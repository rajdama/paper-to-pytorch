"""
Train TinyGPT on Tiny Shakespeare.

Run from the repo root:
    python -m bonus.step13_gpt_decoder_only.train_gpt                       # characters, ~5 min on CPU
    python -m bonus.step13_gpt_decoder_only.train_gpt --tokenizer bpe       # BPE tokens (Step 14)
    python -m bonus.step13_gpt_decoder_only.train_gpt --steps 5000 --d-model 256 --layers 6   # on a GPU

Saves:
    checkpoints/tinygpt_<tokenizer>.pt                    model + tokenizer, used by generate.py
    bonus/step13_gpt_decoder_only/loss_<tokenizer>.png    train / validation loss curves
"""

import argparse
import math
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from bonus.data import ROOT, tiny_shakespeare
from bonus.step13_gpt_decoder_only.char_tokenizer import CharTokenizer
from bonus.step13_gpt_decoder_only.gpt import GPT, GPTConfig
from bonus.step13_gpt_decoder_only.sampling import generate


def build_tokenizer(kind: str, text: str, vocab_size: int):
    if kind == "char":
        return CharTokenizer(text)
    from bonus.step14_bpe_tokenizer.bpe import BPETokenizer
    t0 = time.time()
    tok = BPETokenizer()
    tok.train(text, vocab_size)
    print(f"trained BPE tokenizer: {vocab_size} tokens in {time.time() - t0:.1f}s")
    return tok


def lr_at(step: int, max_lr: float, warmup: int, total: int, min_ratio: float = 0.1) -> float:
    """Linear warmup, then cosine decay to min_ratio * max_lr (what most modern LLMs use)."""
    if step < warmup:
        return max_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return max_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", choices=["char", "bpe"], default="char")
    p.add_argument("--vocab-size", type=int, default=512, help="BPE vocabulary size")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--prompt", default="ROMEO:")
    p.add_argument("--sample-tokens", type=int, default=400)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- data ----
    text = tiny_shakespeare()
    tok = build_tokenizer(args.tokenizer, text, args.vocab_size)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]
    print(f"{len(text):,} characters -> {len(data):,} tokens "
          f"({len(text) / len(data):.2f} chars/token), vocab {tok.vocab_size}")

    def get_batch(split):
        src = train_data if split == "train" else val_data
        ix = torch.randint(len(src) - args.block_size - 1, (args.batch_size,))
        x = torch.stack([src[i:i + args.block_size] for i in ix])
        y = torch.stack([src[i + 1:i + args.block_size + 1] for i in ix])   # next tokens
        return x.to(device), y.to(device)

    # ---- model ----
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=args.block_size, d_model=args.d_model,
                    num_heads=args.heads, num_layers=args.layers, dropout=args.dropout)
    model = GPT(cfg).to(device)
    print(f"TinyGPT: {model.num_parameters():,} parameters on {device}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.1)

    @torch.no_grad()
    def estimate_loss(iters=20):
        model.eval()
        out = {}
        for split in ("train", "val"):
            out[split] = sum(model(*get_batch(split))[1].item() for _ in range(iters)) / iters
        model.train()
        return out

    # ---- train ----
    history = {"step": [], "train": [], "val": []}
    t0 = time.time()
    model.train()
    for step in range(args.steps + 1):
        if step % args.eval_every == 0 or step == args.steps:
            losses = estimate_loss()
            history["step"].append(step)
            history["train"].append(losses["train"])
            history["val"].append(losses["val"])
            print(f"step {step:5d}  train {losses['train']:.3f}  val {losses['val']:.3f}  "
                  f"({time.time() - t0:.0f}s)")
        if step == args.steps:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.lr, args.warmup, args.steps)
        x, y = get_batch("train")
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    # ---- save ----
    ckpt_dir = ROOT / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / f"tinygpt_{args.tokenizer}.pt"
    torch.save({"config": asdict(cfg), "tokenizer": tok.state(), "state_dict": model.state_dict()},
               ckpt_path)
    print(f"saved {ckpt_path}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["step"], history["train"], label="train")
    ax.plot(history["step"], history["val"], label="validation")
    ax.set_xlabel("step"); ax.set_ylabel("cross-entropy loss"); ax.grid(alpha=0.3); ax.legend()
    ax.set_title(f"TinyGPT on Tiny Shakespeare ({args.tokenizer} tokens)")
    fig.tight_layout()
    plot_path = Path(__file__).parent / f"loss_{args.tokenizer}.png"
    fig.savefig(plot_path, dpi=110)
    print(f"saved {plot_path}")

    # ---- sample ----
    prompt = torch.tensor([tok.encode(args.prompt)], dtype=torch.long, device=device)
    g = torch.Generator(device=device).manual_seed(0)
    out = generate(model, prompt, args.sample_tokens, temperature=0.8, top_k=50, generator=g)
    print("\n----- sample (temperature 0.8, top-k 50) -----")
    print(tok.decode(out[0].tolist()))
    print("-----------------------------------------------")


if __name__ == "__main__":
    main()
