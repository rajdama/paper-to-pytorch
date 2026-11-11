"""Run from the repo root:  python -m step10_full_transformer.test_transformer"""

import torch
import torch.nn.functional as F

from step10_full_transformer.transformer import Transformer, count_parameters

torch.manual_seed(0)
PAD = 0


def tiny(**kw):
    cfg = dict(src_vocab_size=20, tgt_vocab_size=20, d_model=32, num_heads=4,
               num_layers=2, d_ff=64, dropout=0.0)
    cfg.update(kw)
    return Transformer(**cfg)


def test_logits_shape():
    model = tiny(tgt_vocab_size=25)
    src = torch.randint(1, 20, (3, 9))
    tgt = torch.randint(1, 25, (3, 6))
    assert model(src, tgt).shape == (3, 6, 25)


def test_output_layer_shares_weights_with_target_embedding():
    model = tiny()
    assert model.generator.weight is model.tgt_embed.embedding.weight


def test_shared_src_tgt_embeddings():
    model = tiny(share_src_tgt_embeddings=True)
    assert model.src_embed is model.tgt_embed


def test_paper_base_model_size():
    """Paper, Table 3: the base model has about 65M parameters (shared ~37k BPE vocab)."""
    model = Transformer(37000, 37000, share_src_tgt_embeddings=True)
    n = count_parameters(model)
    print(f"    base model parameters: {n / 1e6:.1f}M")
    assert 60e6 < n < 66e6


def test_decoder_is_causal_end_to_end():
    model = tiny().eval()
    src = torch.randint(1, 20, (1, 7))
    tgt = torch.randint(1, 20, (1, 6))
    out1 = model(src, tgt)
    tgt2 = tgt.clone()
    tgt2[0, 4] = (tgt[0, 4] % 19) + 1          # change target token 4
    out2 = model(src, tgt2)
    assert torch.allclose(out1[:, :4], out2[:, :4], atol=1e-5)


def test_can_overfit_one_batch():
    """The ultimate sanity check: a working model can memorise a tiny batch."""
    model = tiny()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    src = torch.randint(3, 20, (8, 6))
    tgt = torch.cat([torch.ones(8, 1, dtype=torch.long), src], dim=1)  # <bos>=1 + copy of src
    tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
    losses = []
    for _ in range(150):
        logits = model(src, tgt_in)
        loss = F.cross_entropy(logits.reshape(-1, 20), tgt_out.reshape(-1), ignore_index=PAD)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    print(f"    loss {losses[0]:.2f} -> {losses[-1]:.3f}")
    assert losses[-1] < 0.1 * losses[0]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  ✓ {name}")
    print("Step 10: all tests passed")
