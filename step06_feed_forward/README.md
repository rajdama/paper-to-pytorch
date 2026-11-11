# Step 06 · Position-wise Feed-Forward Network

<p align="center"><img src="diagram.svg" width="420" alt="Transformer diagram with this step's component highlighted"></p>

> 🚧 **Not built yet.** This page describes the plan for this step.

**Paper:** Section 3.3

## The idea

A small two-layer MLP applied to every position separately: $\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$, expanding to $d_{ff} = 2048$ and back to 512.

## Planned

- `PositionwiseFeedForward(d_model, d_ff, dropout)`
- test: each position is processed independently
