"""
Builds the Colab notebooks in notebooks/ (kept as code so they stay easy to review).
Run from the repo root:   python tools/make_notebooks.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = "rajdama/paper-to-pytorch"


def colab_badge(name):
    url = f"https://colab.research.google.com/github/{REPO}/blob/main/notebooks/{name}"
    return f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({url})"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


SETUP = code("""
# Get the code (Colab already has PyTorch and matplotlib installed)
import os
if not os.path.exists("step03_attention"):
    if not os.path.exists("paper-to-pytorch"):
        !git clone -q https://github.com/rajdama/paper-to-pytorch.git
    %cd paper-to-pytorch

import torch
from IPython.display import Image, display
print("PyTorch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU is fine for this notebook)")
""")


def notebook(cells, gpu=False):
    meta = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}, "colab": {"provenance": []}}
    if gpu:
        meta["accelerator"] = "GPU"
        meta["colab"]["gpuType"] = "T4"
    return {"nbformat": 4, "nbformat_minor": 0, "metadata": meta, "cells": cells}


NB1 = "01_build_the_transformer.ipynb"
nb1 = notebook([
    md(f"""
# 1 · Build the Transformer, piece by piece

{colab_badge(NB1)}

This notebook walks through Steps 01–10 of [paper-to-pytorch](https://github.com/{REPO}):
every component of *Attention Is All You Need*, each one tested against the paper's
formula or PyTorch's own implementation. Run the cells top to bottom.
"""),
    SETUP,
    md("""
## Step 01–02 · Embeddings and positional encoding
Tokens become vectors, and every position gets a unique sine/cosine "signature" added on top.
"""),
    code("""
from step01_embeddings import TokenEmbedding
from step02_positional_encoding import PositionalEncoding

emb = TokenEmbedding(vocab_size=100, d_model=16)
pe = PositionalEncoding(d_model=16, dropout=0.0)
tokens = torch.tensor([[5, 42, 7, 7]])           # note: token 7 appears twice
x = pe(emb(tokens))
print("shape:", tuple(x.shape))
print("same token, different positions -> different vectors:", not torch.allclose(x[0, 2], x[0, 3]))

!python -m step02_positional_encoding.demo
display(Image("step02_positional_encoding/pe_heatmap.png"))
"""),
    md("""
## Step 03 · Scaled dot-product attention
$\\text{softmax}(QK^\\top/\\sqrt{d_k})\\,V$. The demo uses hand-made vectors where animals look alike and colours look alike.
"""),
    code("""
!python -m step03_attention.demo
display(Image("step03_attention/attention_heatmap.png"))
"""),
    md("""
**Try it:** why divide by $\\sqrt{d_k}$? Make the vectors 10× bigger and compare the attention with and without scaling.
"""),
    code("""
import torch.nn.functional as F
q = torch.randn(1, 6, 64) * 10
k = torch.randn(1, 6, 64) * 10
for scaled in (True, False):
    scores = q @ k.transpose(-2, -1) / (64 ** 0.5 if scaled else 1)
    w = F.softmax(scores, dim=-1)
    print(f"scaled={scaled!s:5}  largest weight in each row:", w.max(-1).values.squeeze().numpy().round(3))
# Unscaled: rows become one-hot, softmax saturates, and gradients vanish.
"""),
    md("## Step 04–05 · Masks and multi-head attention"),
    code("""
from step04_masking import make_causal_mask
from step05_multi_head_attention import MultiHeadAttention

print(make_causal_mask(5)[0, 0].int())
mha = MultiHeadAttention(d_model=64, num_heads=8)
x = torch.randn(2, 5, 64)
out = mha(x, x, x, make_causal_mask(5))
print("output:", tuple(out.shape), " attention weights per head:", tuple(mha.attn_weights.shape))

!python -m step05_multi_head_attention.demo
display(Image("step05_multi_head_attention/heads.png"))
"""),
    md("## Step 06–09 · Feed-forward, Add & Norm, encoder, decoder"),
    code("""
from step08_encoder import Encoder
from step09_decoder import Decoder

enc = Encoder(num_layers=2, d_model=64, num_heads=8, d_ff=256)
dec = Decoder(num_layers=2, d_model=64, num_heads=8, d_ff=256)
src, tgt = torch.randn(1, 9, 64), torch.randn(1, 5, 64)
memory = enc(src)
print("encoder memory:", tuple(memory.shape))
print("decoder output:", tuple(dec(tgt, memory, tgt_mask=make_causal_mask(5)).shape),
      " <- 5 target tokens reading 9 source tokens through cross-attention")
"""),
    md("""
## Step 10 · The full Transformer
The paper's base configuration: $d_{model}=512$, 8 heads, 6 layers, $d_{ff}=2048$.
"""),
    code("!python -m step10_full_transformer.summary"),
    md("## Run every test in the repo"),
    code("!python -m pytest -q"),
    md("""
**Next:** [2 · Train the Transformer](https://colab.research.google.com/github/rajdama/paper-to-pytorch/blob/main/notebooks/02_train_the_transformer.ipynb)
"""),
])

NB2 = "02_train_the_transformer.ipynb"
nb2 = notebook([
    md(f"""
# 2 · Train the Transformer and watch it think

{colab_badge(NB2)}

Steps 11–12 of [paper-to-pytorch](https://github.com/{REPO}): train the full encoder-decoder
Transformer on a toy translation task (reverse a sequence of digits), decode with greedy
search and beam search, then look inside at what its cross-attention learned.

Runs in about a minute on a CPU; faster on a GPU (*Runtime → Change runtime type → T4 GPU*).
"""),
    SETUP,
    md("## Train (label smoothing + the paper's warmup schedule)"),
    code("""
!python -m step11_training.train_reverse --steps 2000
display(Image("step11_training/training_curves.png"))
"""),
    md("## Decode: greedy vs beam search"),
    code("""
!python -m step12_inference.demo
!python -m step12_inference.demo 3 1 4 1 5 9 2 6
"""),
    md("""
## Look inside: cross-attention
To reverse a sequence, the decoder has to fetch the *last* source digit first. If the model
learned this, cross-attention lights up along the **anti-diagonal**.
"""),
    code("""
!python -m step12_inference.attention_map 2 7 1 8 2 8 1 8
display(Image("step12_inference/cross_attention.png"))
"""),
    md("""
## Experiment: what if we remove positional encoding?
Attention by itself doesn't know token order. Without positions, the encoder can't tell
which digit came first, so reversing should become impossible. Let's check.
"""),
    code("""
import torch.nn as nn
from step10_full_transformer import Transformer
from step11_training import (FIRST_SYMBOL, PAD, LabelSmoothingLoss, make_optimizer,
                             reverse_task_batch)

device = "cuda" if torch.cuda.is_available() else "cpu"

def train(use_positions: bool, steps=1500):
    torch.manual_seed(0)
    model = Transformer(FIRST_SYMBOL + 10, FIRST_SYMBOL + 10, d_model=64, num_heads=4,
                        num_layers=2, d_ff=256, share_src_tgt_embeddings=True).to(device)
    if not use_positions:
        model.positional = nn.Identity()          # remove Step 02 entirely
    crit = LabelSmoothingLoss(0.1)
    opt, sched = make_optimizer(model, 64, warmup=200, factor=0.25)
    for step in range(steps):
        src, tgt = reverse_task_batch(64, device=device)
        logits = model(src, tgt[:, :-1])
        loss = crit(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    model.eval()
    with torch.no_grad():
        src, tgt = reverse_task_batch(500, device=device)
        logits = model(src, tgt[:, :-1])
    keep = tgt[:, 1:] != PAD
    return (logits.argmax(-1)[keep] == tgt[:, 1:][keep]).float().mean().item()

print(f"with positional encoding:    token accuracy {train(True):.1%}")
print(f"without positional encoding: token accuracy {train(False):.1%}")
"""),
    md("""
**Questions to think about**
- Without positions, the accuracy isn't zero. Which tokens can it still get right, and why?
- In `step11_training/train_reverse.py`, try `--warmup 1`. Does the post-norm Transformer still train?

**Next:** [3 · TinyGPT](https://colab.research.google.com/github/rajdama/paper-to-pytorch/blob/main/notebooks/03_tinygpt.ipynb)
"""),
])

NB3 = "03_tinygpt.ipynb"
nb3 = notebook([
    md(f"""
# 3 · TinyGPT: from the Transformer to a language model

{colab_badge(NB3)}

The bonus steps of [paper-to-pytorch](https://github.com/{REPO}): drop the encoder, keep the
masked decoder, and train it to predict the next character of Shakespeare. Then build a
BPE tokenizer and a KV cache.

**Use a GPU:** *Runtime → Change runtime type → T4 GPU*. Training takes about 10 minutes.
"""),
    SETUP,
    md("""
## Train a bigger TinyGPT on the GPU
About 10M parameters, 256 characters of context. (The CI version on GitHub is much smaller
because it runs on a CPU.)
"""),
    code("""
!python -m bonus.step13_gpt_decoder_only.train_gpt \\
    --steps 4000 --d-model 384 --layers 6 --heads 6 --block-size 256 \\
    --batch-size 64 --dropout 0.2 --lr 1e-3 --eval-every 500 --sample-tokens 600
display(Image("bonus/step13_gpt_decoder_only/loss_char.png"))
"""),
    md("""
## Play with sampling
- **temperature** < 1: safer and more repetitive; > 1: more creative and more broken
- **top-k / top-p**: cut off the unlikely tail of the distribution
"""),
    code("""
for temperature in (0.5, 1.0, 1.4):
    print(f"========== temperature {temperature} ==========")
    !python -m bonus.step13_gpt_decoder_only.generate --prompt "JULIET:" --tokens 250 --temperature {temperature}
"""),
    code("""
!python -m bonus.step13_gpt_decoder_only.generate --prompt "KING HENRY:" --tokens 250 --temperature 0.9 --top-p 0.9
"""),
    md("""
## Bonus 14 · BPE tokenizer
Byte Pair Encoding merges the most frequent pairs of tokens over and over. Watch what it learns.
"""),
    code("!python -m bonus.step14_bpe_tokenizer.demo"),
    md("Now train TinyGPT on BPE tokens instead of characters. Each token covers several characters, so the same context window sees much more text."),
    code("""
!python -m bonus.step13_gpt_decoder_only.train_gpt --tokenizer bpe --vocab-size 1024 \\
    --steps 4000 --d-model 384 --layers 6 --heads 6 --block-size 256 \\
    --batch-size 64 --dropout 0.2 --eval-every 500 --sample-tokens 200
"""),
    md("""
> Note: the char and BPE losses can't be compared directly; one is per character, the
> other per token. Compare the samples instead.

## Bonus 15 · KV cache
Without a cache, generating each new token re-processes the whole context. With one, each
step processes a single token.
"""),
    code("""
!python -m bonus.step15_kv_cache.benchmark
display(Image("bonus/step15_kv_cache/benchmark.png"))
"""),
    code("""
# Generate with the cache. Same seed without --kv-cache gives identical text, just slower.
%time !python -m bonus.step13_gpt_decoder_only.generate --prompt "ROMEO:" --tokens 240 --kv-cache
%time !python -m bonus.step13_gpt_decoder_only.generate --prompt "ROMEO:" --tokens 240
"""),
])

if __name__ == "__main__":
    out = ROOT / "notebooks"
    out.mkdir(exist_ok=True)
    for name, nb in ((NB1, nb1), (NB2, nb2), (NB3, nb3)):
        (out / name).write_text(json.dumps(nb, indent=1) + "\n")
        print(f"wrote notebooks/{name}")
