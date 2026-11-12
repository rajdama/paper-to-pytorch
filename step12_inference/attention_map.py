"""
What did the trained Transformer learn? Look at its cross-attention.
First train it:   python -m step11_training.train_reverse
Then run:         python -m step12_inference.attention_map
Saves step12_inference/cross_attention.png

To reverse a sequence, the decoder must fetch the LAST source digit first, then the
second-to-last, and so on. If the model learned the task the way we'd hope, its
cross-attention (decoder queries -> encoder keys) should light up along the
ANTI-diagonal.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from step10_full_transformer.transformer import Transformer
from step11_training.training import BOS, EOS, FIRST_SYMBOL
from step12_inference.decoding import greedy_decode

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoints" / "reverse.pt"


def label(i: int) -> str:
    return {BOS: "<bos>", EOS: "<eos>"}.get(i, str(i - FIRST_SYMBOL))


def main():
    if not CKPT.exists():
        sys.exit("No trained model found. Run:  python -m step11_training.train_reverse")
    ckpt = torch.load(CKPT, map_location="cpu")
    model = Transformer(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    digits = [int(d) for d in sys.argv[1:]] or [3, 1, 4, 1, 5, 9, 2, 6]
    src = torch.tensor([[d + FIRST_SYMBOL for d in digits] + [EOS]])
    out = greedy_decode(model, src, len(digits) + 2, BOS, EOS)

    # run the decoder once more on its own output so every layer stores its attention weights
    with torch.no_grad():
        memory, src_mask = model.encode(src)
        model.decode(out[:, :-1], memory, src_mask)

    layers = model.decoder.layers
    src_labels = [label(i) for i in src[0].tolist()]
    row_labels = [label(i) for i in out[0, 1:].tolist()]      # row t predicts output token t+1
    num_heads = layers[0].cross_attn.num_heads

    fig, axes = plt.subplots(len(layers), num_heads + 1,
                             figsize=(2.6 * (num_heads + 1), 2.7 * len(layers)), squeeze=False)
    for li, layer in enumerate(layers):
        w = layer.cross_attn.attn_weights[0]                   # (heads, tgt_len, src_len)
        panels = [*(w[h] for h in range(num_heads)), w.mean(0)]
        titles = [f"head {h + 1}" for h in range(num_heads)] + ["average"]
        for ax, panel, title in zip(axes[li], panels, titles):
            ax.imshow(panel.numpy(), cmap="Oranges", vmin=0, vmax=1)
            ax.set_title(f"layer {li + 1} · {title}", fontsize=9)
            ax.set_xticks(range(len(src_labels)), src_labels, fontsize=7)
            ax.set_yticks(range(len(row_labels)), row_labels, fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("source token (encoder)", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("output token being written", fontsize=8)
    fig.suptitle(f"Cross-attention while reversing {digits}", fontsize=12)
    fig.tight_layout()
    path = Path(__file__).parent / "cross_attention.png"
    fig.savefig(path, dpi=110)
    print(f"input  {digits}\noutput {[int(x) for x in row_labels if x.isdigit()]}\nsaved {path}")


if __name__ == "__main__":
    main()
