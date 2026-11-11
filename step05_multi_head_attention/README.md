# Step 05 · Multi-Head Attention

<p align="center"><img src="diagram.svg" width="420" alt="Transformer diagram with this step's component highlighted"></p>

> 🚧 **Not built yet.** This page describes the plan for this step.

**Paper:** Section 3.2.2 (Figure 2, right)

## The idea

Instead of one attention, run $h$ smaller ones in parallel, each with its own learned projections $W^Q_i, W^K_i, W^V_i$, then concatenate and project with $W^O$. Different heads can learn different relationships.

## Planned

- `MultiHeadAttention(d_model, num_heads)`, reusing Step 03's function
- test: matches `torch.nn.MultiheadAttention` when given the same weights
- demo: one heatmap per head
