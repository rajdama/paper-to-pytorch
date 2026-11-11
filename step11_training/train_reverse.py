"""
Train a small Transformer to reverse sequences of numbers.
Run from the repo root:  python -m step11_training.train_reverse            (~1-2 min on CPU)
                         python -m step11_training.train_reverse --steps 500 (quick look)

Saves:
  checkpoints/reverse.pt                - the trained model (used by Step 12)
  step11_training/training_curves.png   - loss, accuracy and learning rate
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from step10_full_transformer.transformer import Transformer, count_parameters
from step11_training.training import (FIRST_SYMBOL, PAD, LabelSmoothingLoss,
                                      make_optimizer, reverse_task_batch)

ROOT = Path(__file__).resolve().parent.parent

CONFIG = dict(src_vocab_size=FIRST_SYMBOL + 10, tgt_vocab_size=FIRST_SYMBOL + 10,
              d_model=64, num_heads=4, num_layers=2, d_ff=256, dropout=0.1,
              pad_idx=PAD, share_src_tgt_embeddings=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Transformer(**CONFIG).to(device)
    print(f"model: {count_parameters(model):,} parameters on {device}")

    criterion = LabelSmoothingLoss(smoothing=0.1, ignore_index=PAD)
    # factor 0.25 keeps the peak learning rate sensible for this small d_model
    opt, sched = make_optimizer(model, CONFIG["d_model"], args.warmup, factor=0.25)

    history = {"step": [], "loss": [], "acc": [], "lr": []}
    model.train()
    for step in range(1, args.steps + 1):
        src, tgt = reverse_task_batch(args.batch_size, device=device)
        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]          # teacher forcing: shift by one

        logits = model(src, tgt_in)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 50 == 0 or step == 1:
            keep = tgt_out != PAD
            acc = (logits.argmax(-1)[keep] == tgt_out[keep]).float().mean().item()
            history["step"].append(step); history["loss"].append(loss.item())
            history["acc"].append(acc); history["lr"].append(sched.get_last_lr()[0])
            if step % 250 == 0 or step == 1:
                print(f"step {step:5d}  loss {loss.item():.3f}  token acc {acc:6.1%}  "
                      f"lr {sched.get_last_lr()[0]:.5f}")

    ckpt_dir = ROOT / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    torch.save({"config": CONFIG, "state_dict": model.state_dict()}, ckpt_dir / "reverse.pt")
    print(f"saved {ckpt_dir / 'reverse.pt'}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history["step"], history["loss"]); axes[0].set_title("loss (label-smoothed)")
    axes[1].plot(history["step"], history["acc"]); axes[1].set_title("token accuracy (teacher forcing)")
    axes[2].plot(history["step"], history["lr"]); axes[2].set_title("learning rate: warmup then decay")
    for ax in axes:
        ax.set_xlabel("step"); ax.grid(alpha=0.3)
    fig.tight_layout()
    out = Path(__file__).parent / "training_curves.png"
    fig.savefig(out, dpi=110)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
