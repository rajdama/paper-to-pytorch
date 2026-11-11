# Step 07 · Residual Connection + Layer Norm

<p align="center"><img src="diagram.svg" width="420" alt="Transformer diagram with this step's component highlighted"></p>

> 🚧 **Not built yet.** This page describes the plan for this step.

**Paper:** Section 3.1 and 5.4

## The idea

Every sub-layer is wrapped as $\text{LayerNorm}(x + \text{Sublayer}(x))$. The residual path lets gradients flow through deep stacks; LayerNorm keeps activations stable.

## Planned

- `LayerNorm` from scratch, checked against `nn.LayerNorm`
- `SublayerConnection` (post-norm as in the paper, with a pre-norm option as in GPT)
