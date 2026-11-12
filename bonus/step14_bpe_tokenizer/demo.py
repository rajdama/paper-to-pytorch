"""
Train BPE on Tiny Shakespeare and look at what it learns.
Run from the repo root:  python -m bonus.step14_bpe_tokenizer.demo        (~30 s, pure Python)
"""

import time

from bonus.data import tiny_shakespeare
from bonus.step14_bpe_tokenizer.bpe import BPETokenizer

text = tiny_shakespeare()
n_bytes = len(text.encode("utf-8"))
sentence = "ROMEO: But, soft! what light through yonder window breaks?"

print(f"Tiny Shakespeare: {n_bytes:,} bytes\n")
print(f"{'vocab size':>10} | {'tokens':>9} | {'bytes per token':>15} | train time")
print("-" * 54)
tokenizers = {}
for vocab_size in (256, 512, 1024):
    t0 = time.time()
    tok = BPETokenizer()
    tok.train(text, vocab_size)
    secs = time.time() - t0
    n_tokens = len(tok.encode(text))
    tokenizers[vocab_size] = tok
    print(f"{vocab_size:>10} | {n_tokens:>9,} | {n_bytes / n_tokens:>15.2f} | {secs:5.1f}s")

tok = tokenizers[1024]
print("\nThe first 20 merges it learned (most frequent pairs first):")
for (a, b), new in list(tok.merges.items())[:20]:
    print(f"  {tok.vocab[a]!r:>8} + {tok.vocab[b]!r:<8} -> {tok.vocab[new]!r}")

print("\nThe longest tokens in the 1024 vocabulary:")
longest = sorted(tok.vocab.values(), key=len, reverse=True)[:12]
print("  " + ", ".join(repr(t.decode("utf-8", "replace")) for t in longest))

print(f"\nHow one line is split ({sentence!r}):")
for vocab_size, t in tokenizers.items():
    pieces = t.token_strings(sentence)
    print(f"  vocab {vocab_size:>4} ({len(pieces):2d} tokens): " + "|".join(pieces))
