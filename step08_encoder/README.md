# Step 08 · Encoder

<p align="center"><img src="diagram.svg" width="420" alt="Transformer diagram with this step's component highlighted"></p>

> 🚧 **Not built yet.** This page describes the plan for this step.

**Paper:** Section 3.1

## The idea

One encoder layer = self-attention + feed-forward, each wrapped in Add & Norm. Stack $N = 6$ of them.

## Planned

- `EncoderLayer`, `Encoder` (imports Steps 01-07)
- test: shapes, padding positions don't affect other tokens
