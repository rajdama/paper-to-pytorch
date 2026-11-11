"""Run from the repo root:  python -m step06_feed_forward.test_feed_forward"""

import torch

from step06_feed_forward.feed_forward import PositionwiseFeedForward

torch.manual_seed(0)


def test_shape():
    ffn = PositionwiseFeedForward(d_model=32, d_ff=128)
    assert ffn(torch.randn(2, 9, 32)).shape == (2, 9, 32)


def test_each_position_processed_independently():
    ffn = PositionwiseFeedForward(d_model=16, d_ff=64, dropout=0.0)
    x = torch.randn(1, 5, 16)
    full = ffn(x)
    for i in range(5):
        alone = ffn(x[:, i:i + 1])                 # run position i by itself
        assert torch.allclose(full[:, i:i + 1], alone, atol=1e-6)


def test_matches_formula():
    ffn = PositionwiseFeedForward(d_model=8, d_ff=32, dropout=0.0)
    x = torch.randn(3, 8)
    W1, b1 = ffn.linear1.weight.T, ffn.linear1.bias
    W2, b2 = ffn.linear2.weight.T, ffn.linear2.bias
    expected = torch.clamp(x @ W1 + b1, min=0) @ W2 + b2      # max(0, xW1 + b1)W2 + b2
    assert torch.allclose(ffn(x), expected, atol=1e-6)


def test_paper_parameter_count():
    ffn = PositionwiseFeedForward(d_model=512, d_ff=2048)
    n = sum(p.numel() for p in ffn.parameters())
    assert n == 512 * 2048 + 2048 + 2048 * 512 + 512           # ~2.1M per layer


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  ✓ {name}")
    print("Step 06: all tests passed")
