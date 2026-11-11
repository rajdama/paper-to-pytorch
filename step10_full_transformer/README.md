# Step 10 · The Full Transformer

<p align="center"><img src="diagram.svg" width="420" alt="Transformer diagram with this step's component highlighted"></p>

> 🚧 **Not built yet.** This page describes the plan for this step.

**Paper:** Section 3 (Figure 1)

## The idea

Embeddings + positional encoding + encoder + decoder + final Linear and softmax over the vocabulary, with weight sharing between embeddings and the output layer.

## Planned

- `Transformer` with the paper's base config (d_model 512, 8 heads, 6 layers, d_ff 2048)
- parameter count checked against the paper's ~65M
