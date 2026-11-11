# Step 09 · The Decoder

<p align="center"><img src="diagram.svg" width="420" alt="Transformer diagram with the decoder stack highlighted"></p>

**Paper:** Section 3.1, *Encoder and Decoder Stacks*

## The idea

The decoder writes the output sentence. Each layer (the right-hand box in the
diagram) has **three** sub-layers, each wrapped in Add & Norm:

| # | Sub-layer | Q from | K, V from | Mask |
|:-:|---|---|---|---|
| 1 | Masked self-attention | decoder | decoder | padding + **causal** |
| 2 | **Cross-attention** | decoder | **encoder output** | source padding |
| 3 | Feed-forward | | | |

**Sub-layer 1** lets each target position look at the words written *so far*. The causal
mask (Step 04) stops it from peeking at the answer.

**Sub-layer 2** is where translation actually happens. The decoder asks "which source words
matter for what I'm writing now?" Its queries come from the target side, while keys and
values come from the encoder's memory. That's the arrow labelled **K, V** crossing from
the encoder to the decoder in the diagram. Because the query and key lengths can differ
(5 target words reading 9 source words), Step 03 was written to allow it.

**Sub-layer 3** is the same position-wise feed-forward as in the encoder.

## What's implemented

| File | Contents |
|---|---|
| `decoder.py` | `DecoderLayer`, `Decoder` (N layers) |
| `test_decoder.py` | different source/target lengths, **future target tokens don't leak**, output depends on the source, source padding ignored |

## Run it

```bash
python -m step09_decoder.test_decoder
```

## Try this

- Delete sub-layer 2 (cross-attention) and you have the block used by GPT. That's exactly what bonus Step 13 does.
- After Step 11 trains a model, plot `cross_attn.attn_weights` for a translated sentence. You'll see which source word each output word "looked at".
