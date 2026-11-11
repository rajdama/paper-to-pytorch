"""Run from the repo root:  python -m step08_encoder.test_encoder"""

import torch

from step04_masking.masks import make_padding_mask
from step08_encoder.encoder import Encoder, EncoderLayer

torch.manual_seed(0)
PAD = 0


def test_layer_shape():
    layer = EncoderLayer(d_model=32, num_heads=4, d_ff=64)
    assert layer(torch.randn(2, 7, 32)).shape == (2, 7, 32)


def test_stack_has_n_independent_layers():
    enc = Encoder(num_layers=6, d_model=32, num_heads=4, d_ff=64)
    assert len(enc.layers) == 6
    w0 = enc.layers[0].self_attn.w_q.weight
    w1 = enc.layers[1].self_attn.w_q.weight
    assert w0.data_ptr() != w1.data_ptr() and not torch.equal(w0, w1)   # not shared copies


def test_padding_does_not_leak():
    """Whatever sits in the pad positions must not change the real tokens' outputs."""
    enc = Encoder(num_layers=2, d_model=32, num_heads=4, d_ff=64).eval()   # eval: dropout off
    tokens = torch.tensor([[5, 6, 7, PAD, PAD]])
    mask = make_padding_mask(tokens, PAD)
    x = torch.randn(1, 5, 32)
    out1 = enc(x, mask)
    x2 = x.clone()
    x2[:, 3:] = torch.randn(1, 2, 32) * 100          # garbage in the pad slots
    out2 = enc(x2, mask)
    assert torch.allclose(out1[:, :3], out2[:, :3], atol=1e-5)


def test_every_token_sees_every_token():
    """No causal mask in the encoder: changing the LAST token changes the FIRST output."""
    enc = Encoder(num_layers=1, d_model=32, num_heads=4, d_ff=64).eval()
    x = torch.randn(1, 5, 32)
    x2 = x.clone()
    x2[:, -1] += 1.0
    assert not torch.allclose(enc(x)[:, 0], enc(x2)[:, 0])


def test_pre_norm_variant():
    enc = Encoder(num_layers=2, d_model=32, num_heads=4, d_ff=64, pre_norm=True)
    assert enc(torch.randn(2, 4, 32)).shape == (2, 4, 32)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  ✓ {name}")
    print("Step 08: all tests passed")
