"""Run from the repo root:  python -m step07_add_and_norm.test_add_and_norm"""

import torch
import torch.nn as nn

from step07_add_and_norm.add_and_norm import LayerNorm, SublayerConnection

torch.manual_seed(0)


def test_layernorm_matches_pytorch():
    ours, ref = LayerNorm(16), nn.LayerNorm(16)
    with torch.no_grad():                       # give both the same random gamma/beta
        ours.gamma.copy_(torch.randn(16)); ref.weight.copy_(ours.gamma)
        ours.beta.copy_(torch.randn(16)); ref.bias.copy_(ours.beta)
    x = torch.randn(2, 5, 16) * 7 + 3
    assert torch.allclose(ours(x), ref(x), atol=1e-5)


def test_layernorm_output_statistics():
    x = torch.randn(4, 10, 32) * 50 + 20        # wildly scaled input
    y = LayerNorm(32)(x)
    assert torch.allclose(y.mean(-1), torch.zeros(4, 10), atol=1e-5)
    assert torch.allclose(y.var(-1, unbiased=False), torch.ones(4, 10), atol=1e-3)


def test_post_norm_formula():
    block = SublayerConnection(8, dropout=0.0)
    x = torch.randn(2, 3, 8)
    f = nn.Linear(8, 8)
    assert torch.allclose(block(x, f), block.norm(x + f(x)), atol=1e-6)


def test_pre_norm_formula():
    block = SublayerConnection(8, dropout=0.0, pre_norm=True)
    x = torch.randn(2, 3, 8)
    f = nn.Linear(8, 8)
    assert torch.allclose(block(x, f), x + f(block.norm(x)), atol=1e-6)


def test_residual_lets_gradient_through():
    """Even if the sub-layer outputs zero, gradient still reaches x through the skip path."""
    block = SublayerConnection(8, dropout=0.0, pre_norm=True)
    x = torch.randn(1, 4, 8, requires_grad=True)
    block(x, lambda t: torch.zeros_like(t)).sum().backward()
    assert torch.allclose(x.grad, torch.ones_like(x))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  ✓ {name}")
    print("Step 07: all tests passed")
