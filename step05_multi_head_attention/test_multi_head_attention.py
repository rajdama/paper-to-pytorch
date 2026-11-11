"""Run from the repo root:  python -m step05_multi_head_attention.test_multi_head_attention"""

import torch
import torch.nn as nn

from step04_masking.masks import make_causal_mask
from step05_multi_head_attention.multi_head_attention import MultiHeadAttention

torch.manual_seed(0)


def copy_into_pytorch(ours: MultiHeadAttention, d_model: int, h: int) -> nn.MultiheadAttention:
    """Build torch.nn.MultiheadAttention with exactly our weights."""
    ref = nn.MultiheadAttention(d_model, h, batch_first=True)
    with torch.no_grad():
        ref.in_proj_weight.copy_(torch.cat([ours.w_q.weight, ours.w_k.weight, ours.w_v.weight]))
        ref.in_proj_bias.copy_(torch.cat([ours.w_q.bias, ours.w_k.bias, ours.w_v.bias]))
        ref.out_proj.weight.copy_(ours.w_o.weight)
        ref.out_proj.bias.copy_(ours.w_o.bias)
    return ref


def test_shapes():
    mha = MultiHeadAttention(d_model=32, num_heads=4)
    x = torch.randn(2, 7, 32)
    assert mha(x, x, x).shape == (2, 7, 32)
    assert mha.attn_weights.shape == (2, 4, 7, 7)          # one map per head


def test_cross_attention_lengths():
    mha = MultiHeadAttention(d_model=32, num_heads=4)
    dec, enc = torch.randn(2, 5, 32), torch.randn(2, 9, 32)
    assert mha(dec, enc, enc).shape == (2, 5, 32)          # output follows the query length
    assert mha.attn_weights.shape == (2, 4, 5, 9)


def test_matches_pytorch_self_attention():
    d_model, h = 32, 4
    ours = MultiHeadAttention(d_model, h)
    ref = copy_into_pytorch(ours, d_model, h)
    x = torch.randn(3, 6, d_model)
    expected, _ = ref(x, x, x)
    assert torch.allclose(ours(x, x, x), expected, atol=1e-5)


def test_matches_pytorch_with_causal_mask():
    d_model, h, T = 32, 4, 6
    ours = MultiHeadAttention(d_model, h)
    ref = copy_into_pytorch(ours, d_model, h)
    x = torch.randn(2, T, d_model)
    mask = make_causal_mask(T)
    # careful: PyTorch's attn_mask uses the OPPOSITE convention (True = blocked)
    expected, _ = ref(x, x, x, attn_mask=~mask[0, 0])
    assert torch.allclose(ours(x, x, x, mask), expected, atol=1e-5)


def test_parameter_count():
    # 4 projection matrices of d_model x d_model, plus biases
    mha = MultiHeadAttention(d_model=512, num_heads=8)
    assert sum(p.numel() for p in mha.parameters()) == 4 * (512 * 512 + 512)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  ✓ {name}")
    print("Step 05: all tests passed")
