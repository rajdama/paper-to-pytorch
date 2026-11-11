"""Run from the repo root:  python -m step09_decoder.test_decoder"""

import torch

from step04_masking.masks import make_causal_mask, make_padding_mask
from step09_decoder.decoder import Decoder, DecoderLayer

torch.manual_seed(0)
PAD = 0


def make_decoder(layers=2):
    return Decoder(num_layers=layers, d_model=32, num_heads=4, d_ff=64).eval()


def test_layer_shape_with_different_lengths():
    layer = DecoderLayer(d_model=32, num_heads=4, d_ff=64)
    x, memory = torch.randn(2, 5, 32), torch.randn(2, 9, 32)    # target 5 tokens, source 9
    assert layer(x, memory).shape == (2, 5, 32)


def test_future_target_tokens_do_not_leak():
    """Changing target token 3 must not change the outputs at positions 0-2."""
    dec = make_decoder()
    x, memory = torch.randn(1, 6, 32), torch.randn(1, 4, 32)
    mask = make_causal_mask(6)
    out1 = dec(x, memory, tgt_mask=mask)
    x2 = x.clone()
    x2[:, 3] = torch.randn(32)
    out2 = dec(x2, memory, tgt_mask=mask)
    assert torch.allclose(out1[:, :3], out2[:, :3], atol=1e-5)
    assert not torch.allclose(out1[:, 3], out2[:, 3])


def test_decoder_actually_reads_the_encoder():
    """Different source sentence -> different output (cross-attention works)."""
    dec = make_decoder()
    x = torch.randn(1, 4, 32)
    out1 = dec(x, torch.randn(1, 5, 32))
    out2 = dec(x, torch.randn(1, 5, 32))
    assert not torch.allclose(out1, out2)


def test_source_padding_is_ignored():
    dec = make_decoder()
    src = torch.tensor([[5, 6, PAD, PAD]])
    src_mask = make_padding_mask(src, PAD)
    x, memory = torch.randn(1, 3, 32), torch.randn(1, 4, 32)
    out1 = dec(x, memory, src_mask)
    memory2 = memory.clone()
    memory2[:, 2:] = torch.randn(1, 2, 32) * 100    # garbage where the source has pads
    out2 = dec(x, memory2, src_mask)
    assert torch.allclose(out1, out2, atol=1e-5)


def test_three_sublayers_per_layer():
    layer = DecoderLayer(d_model=32, num_heads=4, d_ff=64)
    assert hasattr(layer, "self_attn") and hasattr(layer, "cross_attn")
    assert layer.self_attn is not layer.cross_attn       # separate weights


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  ✓ {name}")
    print("Step 09: all tests passed")
