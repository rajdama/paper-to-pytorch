"""
Step 11 - Training pieces from the paper
Paper: Section 5 "Training"

- LabelSmoothingLoss  (Section 5.4, epsilon_ls = 0.1)
- noam_lr / make_optimizer  (Section 5.3: Adam + warmup then inverse-sqrt decay)
- reverse_task_batch  (a toy dataset that trains in about a minute on a CPU)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# special tokens used by the toy task (and Step 12)
PAD, BOS, EOS = 0, 1, 2
FIRST_SYMBOL = 3


class LabelSmoothingLoss(nn.Module):
    """
    Cross-entropy where the target isn't 100% on the right word: (1 - eps) goes to the
    right word and eps is spread over the whole vocabulary. The paper notes this hurts
    perplexity but improves accuracy and BLEU, because the model learns to be less
    over-confident.

        loss = (1 - eps) * NLL(correct word) + eps * mean over vocab of (-log p)
    """

    def __init__(self, smoothing: float = 0.1, ignore_index: int = PAD):
        super().__init__()
        self.eps = smoothing
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """logits: (N, vocab)   target: (N,)"""
        log_probs = F.log_softmax(logits, dim=-1)
        nll = -log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)   # usual cross-entropy
        smooth = -log_probs.mean(dim=-1)                                # uniform-target part
        loss = (1 - self.eps) * nll + self.eps * smooth
        keep = target != self.ignore_index                              # don't learn on <pad>
        return loss[keep].mean()


def noam_lr(step: int, d_model: int, warmup: int, factor: float = 1.0) -> float:
    """lrate = d_model^-0.5 * min(step^-0.5, step * warmup^-1.5)   (Section 5.3)"""
    step = max(step, 1)
    return factor * d_model ** -0.5 * min(step ** -0.5, step * warmup ** -1.5)


def make_optimizer(model: nn.Module, d_model: int, warmup: int = 4000, factor: float = 1.0):
    """Adam with the paper's betas and epsilon, and the warmup schedule."""
    opt = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    # LambdaLR multiplies the base lr (1.0) by our function -> lr = noam_lr(step)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: noam_lr(s + 1, d_model, warmup, factor))
    return opt, sched


def reverse_task_batch(batch_size: int, num_symbols: int = 10, min_len: int = 3,
                       max_len: int = 10, device=None):
    """
    Toy "translation": the target is the source reversed.
        src: [7, 4, 9, EOS]          tgt: [BOS, 9, 4, 7, EOS]
    Sequences have random lengths, so padding and masks get exercised too.
    """
    lengths = torch.randint(min_len, max_len + 1, (batch_size,))
    src = torch.full((batch_size, max_len + 1), PAD, dtype=torch.long)
    tgt = torch.full((batch_size, max_len + 2), PAD, dtype=torch.long)
    for i, n in enumerate(lengths.tolist()):
        seq = torch.randint(FIRST_SYMBOL, FIRST_SYMBOL + num_symbols, (n,))
        src[i, :n], src[i, n] = seq, EOS
        tgt[i, 0], tgt[i, 1:n + 1], tgt[i, n + 1] = BOS, seq.flip(0), EOS
    return src.to(device), tgt.to(device)
