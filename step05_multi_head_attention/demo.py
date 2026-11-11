"""
One attention map per head.
Run from the repo root:  python -m step05_multi_head_attention.demo
Saves step05_multi_head_attention/heads.png

The weights are random (untrained), so the patterns are random too, but notice that
every head already produces a DIFFERENT pattern from the same input. After training,
heads specialise (e.g. one tracks the previous word, another matches verbs to subjects).
"""

from pathlib import Path
import torch
import matplotlib.pyplot as plt

from step04_masking.masks import make_causal_mask
from step05_multi_head_attention.multi_head_attention import MultiHeadAttention

torch.manual_seed(1)
words = ["the", "cat", "sat", "on", "the", "mat"]
d_model, h = 64, 4

x = torch.randn(1, len(words), d_model) * 2
mha = MultiHeadAttention(d_model, h)
with torch.no_grad():
    mha(x, x, x, make_causal_mask(len(words)))

fig, axes = plt.subplots(1, h, figsize=(4 * h, 4))
for i, ax in enumerate(axes):
    ax.imshow(mha.attn_weights[0, i].numpy(), cmap="Blues", vmin=0, vmax=1)
    ax.set_title(f"head {i + 1}")
    ax.set_xticks(range(len(words)), words, rotation=45)
    ax.set_yticks(range(len(words)), words)
fig.suptitle("Same input, different heads, different attention (causal mask on)")
fig.tight_layout()
path = Path(__file__).parent / "heads.png"
fig.savefig(path, dpi=120)
print(f"saved {path}")
