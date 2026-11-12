"""
Where does the trained ViT look? Attention rollout on test digits.
First train:  python -m vit.step03_train_mnist.train_mnist
Then run:     python -m vit.step03_train_mnist.attention_map
Saves vit/step03_train_mnist/attention_rollout.png
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from vit.step02_vit_model.vit import VisionTransformer, attention_rollout
from vit.step03_train_mnist.train_mnist import MNIST_MEAN, MNIST_STD, ROOT, mnist_loaders

CKPT = ROOT / "checkpoints" / "vit_mnist.pt"


def main():
    if not CKPT.exists():
        sys.exit("No trained model found. Run:  python -m vit.step03_train_mnist.train_mnist")
    ckpt = torch.load(CKPT, map_location="cpu")
    model = VisionTransformer(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    _, test_loader = mnist_loaders(64)
    x, y = next(iter(test_loader))
    x, y = x[:8], y[:8]
    with torch.no_grad():
        probs = model(x).softmax(-1)
    rollout = attention_rollout(model.attention_maps())                 # (8, 16)
    grid = int(rollout.size(1) ** 0.5)
    heat = rollout.reshape(-1, 1, grid, grid)
    heat = F.interpolate(heat, size=x.shape[-2:], mode="nearest")[:, 0]
    heat = heat / heat.amax(dim=(1, 2), keepdim=True)

    fig, axes = plt.subplots(2, 8, figsize=(16, 4.4))
    for i in range(8):
        img = (x[i, 0] * MNIST_STD + MNIST_MEAN).numpy()
        pred = probs[i].argmax().item()
        axes[0, i].imshow(img, cmap="gray")
        axes[0, i].set_title(f"label {y[i].item()} · pred {pred} ({probs[i, pred]:.0%})", fontsize=9)
        axes[1, i].imshow(img, cmap="gray")
        axes[1, i].imshow(heat[i].numpy(), cmap="inferno", alpha=0.6)
        for k in range(1, grid):                                         # patch grid lines
            axes[1, i].axhline(k * 7 - 0.5, color="white", lw=0.4, alpha=0.5)
            axes[1, i].axvline(k * 7 - 0.5, color="white", lw=0.4, alpha=0.5)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    axes[0, 0].set_ylabel("input", fontsize=10)
    axes[1, 0].set_ylabel("[CLS] attention\nrollout", fontsize=10)
    fig.suptitle(f"ViT on MNIST (test accuracy {ckpt['test_acc']:.2%}): which 7x7 patches feed the [CLS] token",
                 fontsize=12)
    fig.tight_layout()
    out = Path(__file__).parent / "attention_rollout.png"
    fig.savefig(out, dpi=110)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
