"""
See attention in action on a tiny hand-made example.
Run from the repo root:  python -m step03_attention.demo
Saves step03_attention/attention_heatmap.png

We give 6 "words" hand-made 4-d vectors where animals look alike and colours look
alike. With Q = K = V = those vectors (no learned weights yet), each word
should attend mostly to words of its own kind.
"""

from pathlib import Path
import torch
import matplotlib.pyplot as plt

from step03_attention.attention import scaled_dot_product_attention

words = ["cat", "dog", "red", "lion", "blue", "green"]
#                    animal-ness  colour-ness  noise
vectors = torch.tensor([
    [3.0, 0.0, 0.2, 0.1],   # cat
    [2.8, 0.1, 0.0, 0.3],   # dog
    [0.0, 3.0, 0.3, 0.0],   # red
    [3.1, 0.2, 0.1, 0.0],   # lion
    [0.1, 2.9, 0.0, 0.2],   # blue
    [0.0, 3.1, 0.2, 0.1],   # green
]).unsqueeze(0)             # add batch dim -> (1, 6, 4)

out, weights = scaled_dot_product_attention(vectors, vectors, vectors)
w = weights[0]

for i, word in enumerate(words):
    top = w[i].topk(3)
    pretty = ", ".join(f"{words[j]} {p:.2f}" for p, j in zip(top.values, top.indices))
    print(f"{word:>6} attends to -> {pretty}")

fig, ax = plt.subplots(figsize=(5.5, 4.5))
im = ax.imshow(w.numpy(), cmap="Blues")
ax.set_xticks(range(6), words)
ax.set_yticks(range(6), words)
ax.set_xlabel("key (being looked at)")
ax.set_ylabel("query (doing the looking)")
ax.set_title("Attention weights")
for i in range(6):
    for j in range(6):
        ax.text(j, i, f"{w[i, j]:.2f}", ha="center", va="center", fontsize=8,
                color="white" if w[i, j] > 0.5 else "black")
fig.colorbar(im)
fig.tight_layout()
path = Path(__file__).parent / "attention_heatmap.png"
fig.savefig(path, dpi=120)
print(f"saved {path}")
