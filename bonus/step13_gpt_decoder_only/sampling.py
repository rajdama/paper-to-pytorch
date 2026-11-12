"""
Turning next-token logits into text.

    temperature  - divide logits before softmax: < 1 sharper (safer), > 1 flatter (wilder),
                   0 means greedy (always the top token)
    top_k        - keep only the k most likely tokens
    top_p        - "nucleus" sampling: keep the smallest set of tokens whose
                   probabilities add up to at least p (Holtzman et al., 2019)
"""

import torch
import torch.nn.functional as F


def filter_logits(logits: torch.Tensor, top_k: int | None = None,
                  top_p: float | None = None) -> torch.Tensor:
    """logits: (B, vocab). Returns a copy where filtered-out tokens are -inf."""
    logits = logits.clone()
    if top_k is not None and top_k < logits.size(-1):
        kth_best = logits.topk(top_k, dim=-1).values[:, -1:]           # (B, 1)
        logits[logits < kth_best] = float("-inf")
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        mass_before = probs.cumsum(dim=-1) - probs                      # probability of everything ranked above
        remove_sorted = mass_before >= top_p                            # already covered p without this token
        remove = remove_sorted.scatter(-1, sorted_idx, remove_sorted)   # back to vocabulary order
        logits[remove] = float("-inf")
    return logits


def sample_next(logits: torch.Tensor, temperature: float = 1.0, top_k: int | None = None,
                top_p: float | None = None, generator: torch.Generator | None = None) -> torch.Tensor:
    """logits: (B, vocab) for the last position -> next token ids (B, 1)"""
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = filter_logits(logits / temperature, top_k, top_p)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)


@torch.no_grad()
def generate(model, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
             top_k: int | None = None, top_p: float | None = None,
             generator: torch.Generator | None = None) -> torch.Tensor:
    """
    Autoregressive generation WITHOUT a cache: every step re-runs the model on the
    whole context. Simple and correct, but wasteful - bonus Step 15 fixes that.
    idx: (B, T) prompt token ids -> (B, T + max_new_tokens)
    """
    model.eval()
    block_size = model.cfg.block_size
    for _ in range(max_new_tokens):
        context = idx[:, -block_size:]                 # the model can only see block_size tokens
        logits, _ = model(context)
        next_tok = sample_next(logits[:, -1], temperature, top_k, top_p, generator)
        idx = torch.cat([idx, next_tok], dim=1)
    return idx
