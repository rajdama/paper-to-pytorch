"""
ViT Step 03 - Train a small Vision Transformer on MNIST handwritten digits.

Run from the repo root (needs torchvision for the dataset):
    python -m vit.step03_train_mnist.train_mnist                # ~3 epochs, a few minutes on CPU
    python -m vit.step03_train_mnist.train_mnist --epochs 10    # better, on a GPU

Saves:
    checkpoints/vit_mnist.pt
    vit/step03_train_mnist/training_curves.png
"""

import argparse
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from vit.step02_vit_model.vit import VisionTransformer

ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG = dict(image_size=28, patch_size=7, in_channels=1, num_classes=10,
              d_model=64, num_heads=4, num_layers=4, d_ff=128, dropout=0.1)
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081


def mnist_loaders(batch_size: int):
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))])
    train = datasets.MNIST(ROOT / "data", train=True, download=True, transform=tf)
    test = datasets.MNIST(ROOT / "data", train=False, download=True, transform=tf)
    return (torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2),
            torch.utils.data.DataLoader(test, batch_size=512, num_workers=2))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(-1) == y).sum().item()
        total += y.numel()
    model.train()
    return correct / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, test_loader = mnist_loaders(args.batch_size)

    model = VisionTransformer(**CONFIG).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ViT: {n_params:,} parameters on {device}; 28x28 image -> 16 patches of 7x7 + [CLS]")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    total_steps = args.epochs * len(train_loader)
    warmup = len(train_loader) // 4
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warmup,
                                              0.5 * (1 + math.cos(math.pi * s / total_steps))))

    history = {"step": [], "loss": [], "epoch": [], "test_acc": []}
    step, t0 = 0, time.time()
    for epoch in range(1, args.epochs + 1):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y, label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            if step % 50 == 0:
                history["step"].append(step); history["loss"].append(loss.item())
        acc = evaluate(model, test_loader, device)
        history["epoch"].append(epoch); history["test_acc"].append(acc)
        print(f"epoch {epoch}  last loss {loss.item():.3f}  test accuracy {acc:.2%}  ({time.time() - t0:.0f}s)")

    ckpt_dir = ROOT / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    torch.save({"config": CONFIG, "state_dict": model.state_dict(), "test_acc": acc},
               ckpt_dir / "vit_mnist.pt")
    print(f"saved {ckpt_dir / 'vit_mnist.pt'}")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    a1.plot(history["step"], history["loss"]); a1.set_title("training loss (label-smoothed)")
    a1.set_xlabel("step"); a1.grid(alpha=0.3)
    a2.plot(history["epoch"], [100 * a for a in history["test_acc"]], "o-")
    a2.set_title("test accuracy (%)"); a2.set_xlabel("epoch"); a2.grid(alpha=0.3)
    a2.set_xticks(history["epoch"])
    fig.tight_layout()
    out = Path(__file__).parent / "training_curves.png"
    fig.savefig(out, dpi=110)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
