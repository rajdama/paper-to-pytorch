"""
Where do the parameters live? Builds the paper's base model and breaks it down.
Run from the repo root:  python -m step10_full_transformer.summary
"""

from step10_full_transformer.transformer import Transformer, count_parameters

model = Transformer(src_vocab_size=37000, tgt_vocab_size=37000, share_src_tgt_embeddings=True)

enc_layer = model.encoder.layers[0]
dec_layer = model.decoder.layers[0]
rows = [
    ("Shared embedding (src = tgt = output Linear)", count_parameters(model.src_embed)),
    ("Output Linear bias", model.generator.bias.numel()),
    ("Encoder: 6 layers", count_parameters(model.encoder)),
    ("   one layer: self-attention", count_parameters(enc_layer.self_attn)),
    ("   one layer: feed-forward", count_parameters(enc_layer.feed_forward)),
    ("Decoder: 6 layers", count_parameters(model.decoder)),
    ("   one layer: self-attention", count_parameters(dec_layer.self_attn)),
    ("   one layer: cross-attention", count_parameters(dec_layer.cross_attn)),
    ("   one layer: feed-forward", count_parameters(dec_layer.feed_forward)),
]
width = max(len(r[0]) for r in rows) + 2
for name, n in rows:
    print(f"{name:<{width}}{n:>13,}")
print("-" * (width + 13))
print(f"{'TOTAL':<{width}}{count_parameters(model):>13,}   (paper: ~65M)")
