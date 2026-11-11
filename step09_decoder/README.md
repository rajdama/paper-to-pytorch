# Step 09 · Decoder

<p align="center"><img src="diagram.svg" width="420" alt="Transformer diagram with this step's component highlighted"></p>

> 🚧 **Not built yet.** This page describes the plan for this step.

**Paper:** Section 3.1

## The idea

One decoder layer = **masked** self-attention, then **cross-attention** (queries from the decoder, keys and values from the encoder output), then feed-forward. Stack $N = 6$.

## Planned

- `DecoderLayer`, `Decoder`
- test: changing a future target token doesn't change earlier outputs
