"""
Step 12 - Inference: generating the output one token at a time
Paper: Section 6.1 (beam search, beam size 4, length penalty alpha = 0.6)

Training sees the whole target at once (teacher forcing). At inference there is no
target: we start from <bos>, predict one token, append it, and repeat until <eos>.
The encoder runs ONCE; the decoder runs once per generated token.
"""

import torch
import torch.nn.functional as F


@torch.no_grad()
def greedy_decode(model, src: torch.Tensor, max_len: int, bos_idx: int, eos_idx: int):
    """
    Always pick the single most likely next token.
    src: (B, src_len)  ->  returns (B, <= max_len) token ids starting with <bos>
    """
    model.eval()
    memory, src_mask = model.encode(src)                     # encoder: once
    ys = torch.full((src.size(0), 1), bos_idx, dtype=torch.long, device=src.device)
    finished = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)

    for _ in range(max_len - 1):
        logits = model.decode(ys, memory, src_mask)[:, -1]   # only the last position matters
        next_tok = logits.argmax(dim=-1)
        next_tok = next_tok.masked_fill(finished, model.pad_idx)   # done sequences get <pad>
        ys = torch.cat([ys, next_tok.unsqueeze(1)], dim=1)
        finished |= next_tok == eos_idx
        if finished.all():
            break
    return ys


def length_penalty(length: int, alpha: float) -> float:
    """From Wu et al. 2016 (GNMT), used by the paper: without it, beam search prefers short outputs."""
    return ((5 + length) / 6) ** alpha


@torch.no_grad()
def beam_search(model, src: torch.Tensor, max_len: int, bos_idx: int, eos_idx: int,
                beam_size: int = 4, alpha: float = 0.6):
    """
    Keep the `beam_size` most likely partial outputs instead of just one.
    src: (1, src_len), one sentence at a time to keep the code readable
    returns: (1, length) best token ids, starting with <bos>
    """
    model.eval()
    memory, src_mask = model.encode(src)
    beams = [([bos_idx], 0.0)]          # (tokens, total log-probability)
    finished = []

    for _ in range(max_len - 1):
        ys = torch.tensor([toks for toks, _ in beams], device=src.device)
        k = len(beams)
        logits = model.decode(ys, memory.expand(k, -1, -1), src_mask.expand(k, -1, -1, -1))
        log_probs = F.log_softmax(logits[:, -1], dim=-1)          # (k, vocab)

        # every beam proposes its top `beam_size` continuations
        candidates = []
        for i, (toks, score) in enumerate(beams):
            top = log_probs[i].topk(beam_size)
            for lp, tok in zip(top.values.tolist(), top.indices.tolist()):
                candidates.append((toks + [tok], score + lp))
        candidates.sort(key=lambda c: c[1], reverse=True)

        # keep the best; candidates ending in <eos> are complete
        beams = []
        for toks, score in candidates:
            if toks[-1] == eos_idx:
                finished.append((toks, score))
            else:
                beams.append((toks, score))
            if len(beams) == beam_size:
                break
        if not beams or len(finished) >= beam_size:
            break

    pool = finished or beams
    best, _ = max(pool, key=lambda c: c[1] / length_penalty(len(c[0]) - 1, alpha))
    return torch.tensor([best], device=src.device)
