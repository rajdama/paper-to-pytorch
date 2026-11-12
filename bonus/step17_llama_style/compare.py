"""
GPT-2 style vs LLaMA style: same data, same size, same training budget.
Run from the repo root:  python -m bonus.step17_llama_style.compare            (~10 min on CPU)
                         python -m bonus.step17_llama_style.compare --steps 600   (quick look)
Saves bonus/step17_llama_style/compare.png
"""

import argparse
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from bonus.data import tiny_shakespeare
from bonus.step13_gpt_decoder_only.char_tokenizer import CharTokenizer
from bonus.step13_gpt_decoder_only.gpt import GPT, GPTConfig
from bonus.step13_gpt_decoder_only.sampling import generate
from bonus.step17_llama_style.llama import TinyLlama


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eval-every", type=int, default=250)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    text = tiny_shakespeare()
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    splits = {"train": data[:n], "val": data[n:]}
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=args.block_size, d_model=128,
                    num_heads=4, num_layers=4, dropout=0.1)

    def get_batch(split, g):
        src = splits[split]
        ix = torch.randint(len(src) - args.block_size - 1, (args.batch_size,), generator=g)
        x = torch.stack([src[i:i + args.block_size] for i in ix])
        y = torch.stack([src[i + 1:i + args.block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    results = {}
    for name, cls in (("GPT-2 style (LayerNorm, learned positions, ReLU MLP)", GPT),
                      ("LLaMA style (RMSNorm, RoPE, SwiGLU)", TinyLlama)):
        torch.manual_seed(0)
        model = cls(cfg).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.1)
        g_train = torch.Generator().manual_seed(1)                     # identical batches for both
        steps, vals = [], []
        t0 = time.time()
        print(f"\n{name}: {model.num_parameters():,} parameters")
        for step in range(args.steps + 1):
            if step % args.eval_every == 0 or step == args.steps:
                model.eval()
                g_val = torch.Generator().manual_seed(2)
                with torch.no_grad():
                    v = sum(model(*get_batch("val", g_val))[1].item() for _ in range(20)) / 20
                model.train()
                steps.append(step); vals.append(v)
                print(f"  step {step:5d}  val loss {v:.3f}  ({time.time() - t0:.0f}s)")
            if step == args.steps:
                break
            warm = 100
            lr = args.lr * ((step + 1) / warm if step < warm else
                            0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - warm) / (args.steps - warm))))
            for group in opt.param_groups:
                group["lr"] = lr
            _, loss = model(*get_batch("train", g_train))
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sample = generate(model, torch.tensor([tok.encode("ROMEO:")], device=device), 150,
                          temperature=0.8, top_k=50, generator=torch.Generator(device=device).manual_seed(0))
        results[name] = (steps, vals, time.time() - t0, tok.decode(sample[0].tolist()))

    print("\n================ summary ================")
    for name, (_, vals, secs, sample) in results.items():
        print(f"{name}\n  final val loss {vals[-1]:.3f}   training time {secs:.0f}s\n  sample: {sample!r}\n")

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for name, (steps, vals, _, _) in results.items():
        ax.plot(steps, vals, "o-", ms=3, label=name)
    ax.set_xlabel("training step"); ax.set_ylabel("validation loss (nats per character)")
    ax.set_title("Same size, same data, same budget"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_ylim(top=min(3.0, ax.get_ylim()[1]))
    fig.tight_layout()
    out = Path(__file__).parent / "compare.png"
    fig.savefig(out, dpi=110)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
