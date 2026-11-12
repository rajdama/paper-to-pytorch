"""
How much faster is generation with a KV cache?
Run from the repo root:  python -m bonus.step15_kv_cache.benchmark
Saves bonus/step15_kv_cache/benchmark.png

Uses a randomly initialised TinyGPT (speed doesn't depend on training) with greedy
decoding, and checks both methods produce exactly the same tokens.
"""

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from bonus.step13_gpt_decoder_only.gpt import GPT, GPTConfig
from bonus.step13_gpt_decoder_only.sampling import generate
from bonus.step15_kv_cache.kv_cache import generate_with_kv_cache

torch.manual_seed(0)

cfg = GPTConfig(vocab_size=512, block_size=512, d_model=256, num_heads=8, num_layers=6, dropout=0.0)
model = GPT(cfg).eval()
prompt = torch.randint(0, cfg.vocab_size, (1, 16))
print(f"model: {model.num_parameters():,} parameters, prompt 16 tokens, greedy decoding\n")


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


generate_with_kv_cache(model, prompt, 8, temperature=0)          # warm-up
lengths = [64, 128, 256, 448]
rows = []
print(f"{'new tokens':>10} | {'no cache':>9} | {'KV cache':>9} | speedup | tokens/s (cache)")
print("-" * 62)
for n in lengths:
    slow_out, slow = timed(lambda: generate(model, prompt, n, temperature=0))
    fast_out, fast = timed(lambda: generate_with_kv_cache(model, prompt, n, temperature=0))
    assert torch.equal(slow_out, fast_out), "cached and uncached outputs differ!"
    rows.append((n, slow, fast))
    print(f"{n:>10} | {slow:>8.2f}s | {fast:>8.2f}s | {slow / fast:>6.1f}x | {n / fast:>8.0f}")

print("\nidentical outputs in every run ✓")

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot([r[0] for r in rows], [r[1] for r in rows], "o-", label="no cache (recompute everything)")
ax.plot([r[0] for r in rows], [r[2] for r in rows], "o-", label="KV cache (one token per step)")
ax.set_xlabel("tokens generated"); ax.set_ylabel("seconds"); ax.grid(alpha=0.3); ax.legend()
ax.set_title("Generation time: quadratic vs linear")
fig.tight_layout()
out = Path(__file__).parent / "benchmark.png"
fig.savefig(out, dpi=110)
print(f"saved {out}")
