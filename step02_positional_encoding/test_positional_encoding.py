"""Run from the repo root:  python -m step02_positional_encoding.test_positional_encoding"""

import math
import torch

from step02_positional_encoding.positional_encoding import PositionalEncoding, sinusoidal_table


def test_table_shape_and_range():
    pe = sinusoidal_table(max_len=100, d_model=64)
    assert pe.shape == (100, 64)
    assert pe.min() >= -1 and pe.max() <= 1


def test_position_zero():
    # sin(0) = 0 in even dims, cos(0) = 1 in odd dims
    pe = sinusoidal_table(max_len=10, d_model=8)
    assert torch.allclose(pe[0, 0::2], torch.zeros(4))
    assert torch.allclose(pe[0, 1::2], torch.ones(4))


def test_matches_paper_formula():
    d_model, pe = 16, sinusoidal_table(max_len=50, d_model=16)
    for pos in (1, 7, 42):
        for i in range(d_model // 2):
            angle = pos / (10000 ** (2 * i / d_model))
            assert math.isclose(pe[pos, 2 * i].item(), math.sin(angle), abs_tol=1e-5)
            assert math.isclose(pe[pos, 2 * i + 1].item(), math.cos(angle), abs_tol=1e-5)


def test_every_position_is_unique():
    pe = sinusoidal_table(max_len=200, d_model=32)
    dists = torch.cdist(pe, pe) + torch.eye(200) * 1e9   # ignore self-distance
    assert dists.min() > 1e-3


def test_module_adds_encoding():
    module = PositionalEncoding(d_model=32, dropout=0.0)
    x = torch.zeros(2, 10, 32)
    out = module(x)
    assert out.shape == (2, 10, 32)
    assert torch.allclose(out[0], module.pe[:10])        # zeros + PE = PE
    assert not any(p.requires_grad for p in module.parameters())  # nothing to train


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  ✓ {name}")
    print("Step 02: all tests passed")
