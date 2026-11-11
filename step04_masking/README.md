# Step 04 · Masking

<p align="center"><img src="diagram.svg" width="420" alt="Transformer diagram with this step's component highlighted"></p>

> 🚧 **Not built yet.** This page describes the plan for this step.

**Paper:** Section 3.2.3

## The idea

Two masks: a **padding mask** so tokens ignore `<pad>` filler, and a **causal (look-ahead) mask** so the decoder can't peek at future tokens. The paper describes this as setting illegal connections to $-\infty$ before the softmax.

## Planned

- `make_padding_mask(tokens, pad_idx)`
- `make_causal_mask(seq_len)`
- tests: no future leakage, padding ignored, what happens to a fully masked row
