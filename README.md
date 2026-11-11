# paper-to-pytorch

**Implementing every component of *Attention Is All You Need* from scratch in PyTorch, one folder per block of the architecture.**

Each folder builds one piece of the Transformer, shows exactly where that piece sits in
the architecture, tests it, and explains it. Later folders import earlier ones, so by
Step 10 the folders combine into the complete model, which is then trained and used.

<p align="center"><img src="assets/architecture.svg" width="480" alt="The Transformer architecture"></p>

## Roadmap

| | Step | Component | Paper section |
|:-:|---|---|---|
| ✅ | [01](step01_embeddings) | Token embeddings | 3.4 |
| ✅ | [02](step02_positional_encoding) | Positional encoding | 3.5 |
| ✅ | [03](step03_attention) | Scaled dot-product attention | 3.2.1 |
| ⬜ | [04](step04_masking) | Padding & causal masks | 3.2.3 |
| ⬜ | [05](step05_multi_head_attention) | Multi-head attention | 3.2.2 |
| ⬜ | [06](step06_feed_forward) | Position-wise feed-forward | 3.3 |
| ⬜ | [07](step07_add_and_norm) | Residual + layer norm | 3.1 |
| ⬜ | [08](step08_encoder) | Encoder layer & stack | 3.1 |
| ⬜ | [09](step09_decoder) | Decoder layer & stack (with cross-attention) | 3.1 |
| ⬜ | [10](step10_full_transformer) | The full Transformer | 3 |
| ⬜ | [11](step11_training) | Training (label smoothing, warmup schedule) | 5 |
| ⬜ | [12](step12_inference) | Greedy decoding & beam search | 6.1 |

**Bonus:** from the Transformer to modern LLMs

| | Step | Component |
|:-:|---|---|
| ⬜ | [13](bonus/step13_gpt_decoder_only) | TinyGPT: a decoder-only language model |
| ⬜ | [14](bonus/step14_bpe_tokenizer) | Byte Pair Encoding tokenizer |
| ⬜ | [15](bonus/step15_kv_cache) | KV cache for fast generation |

## What's in each step folder

```
stepNN_component/
├── README.md          what it does, the equation, where it sits in the diagram
├── diagram.svg        the architecture with this component highlighted
├── <component>.py     the implementation
├── test_<component>.py  shape checks + comparison with PyTorch's built-ins
└── demo.py            (some steps) a visualisation you can run
```

## Getting started

```bash
git clone https://github.com/<your-username>/paper-to-pytorch.git
cd paper-to-pytorch
pip install -r requirements.txt   # Python 3.10+

# run every test
pytest

# or one step at a time (always from the repo root)
python -m step03_attention.test_attention
python -m step03_attention.demo
```

After editing `tools/make_diagrams.py`, regenerate all diagrams with `python tools/make_diagrams.py`.

## Reference

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017).
*Attention Is All You Need.* NeurIPS. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

The architecture diagrams in this repo are redrawn after Figure 1 of the paper.
