"""Run from the repo root:  python -m bonus.step17_llama_style.test_llama"""

import torch

from bonus.step13_gpt_decoder_only.gpt import GPT, GPTConfig
from bonus.step13_gpt_decoder_only.sampling import generate
from bonus.step17_llama_style.llama import (RMSNorm, SwiGLU, TinyLlama, apply_rope, rope_cache,
                                            swiglu_hidden)

torch.manual_seed(0)


def cfg(**kw):
    c = dict(vocab_size=40, block_size=16, d_model=64, num_heads=4, num_layers=2, dropout=0.0)
    c.update(kw)
    return GPTConfig(**c)


def test_rmsnorm_formula():
    norm = RMSNorm(8)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(8))
    x = torch.randn(3, 8) * 5
    expected = x / torch.sqrt((x ** 2).mean(-1, keepdim=True) + 1e-6) * norm.weight
    assert torch.allclose(norm(x), expected, atol=1e-5)


def test_rmsnorm_output_has_unit_rms():
    y = RMSNorm(32)(torch.randn(4, 32) * 100 + 7)
    assert torch.allclose(y.pow(2).mean(-1).sqrt(), torch.ones(4), atol=1e-3)


def test_rope_position_zero_is_identity():
    cos, sin = rope_cache(10, 16)
    x = torch.randn(16)
    assert torch.allclose(apply_rope(x, cos[0], sin[0]), x)


def test_rope_preserves_length():
    cos, sin = rope_cache(50, 16)
    x = torch.randn(50, 16)
    assert torch.allclose(apply_rope(x, cos, sin).norm(dim=-1), x.norm(dim=-1), atol=1e-5)


def test_rope_depends_only_on_relative_position():
    """The key property: q at position m and k at position n give the same score as at m+s, n+s."""
    cos, sin = rope_cache(100, 16)
    q, k = torch.randn(16), torch.randn(16)
    score = lambda m, n: (apply_rope(q, cos[m], sin[m]) @ apply_rope(k, cos[n], sin[n])).item()
    for shift in (1, 17, 60):
        assert abs(score(9, 4) - score(9 + shift, 4 + shift)) < 1e-4
    assert abs(score(9, 4) - score(9, 6)) > 1e-3                 # different distance, different score


def test_swiglu_matches_mlp_parameter_budget():
    d = 128
    ffn = SwiGLU(d, swiglu_hidden(d))
    n_swiglu = sum(p.numel() for p in ffn.parameters())
    n_mlp = 2 * d * 4 * d                                        # ReLU MLP weights, 4x expansion
    assert abs(n_swiglu - n_mlp) / n_mlp < 0.05


def test_logits_and_causality():
    model = TinyLlama(cfg()).eval()
    idx = torch.randint(0, 40, (2, 12))
    logits, _ = model(idx)
    assert logits.shape == (2, 12, 40)
    idx2 = idx.clone()
    idx2[:, 7:] = (idx2[:, 7:] + 1) % 40
    logits2, _ = model(idx2)
    assert torch.allclose(logits[:, :7], logits2[:, :7], atol=1e-5)


def test_similar_size_to_tinygpt():
    c = cfg(d_model=128, num_layers=4, vocab_size=65, block_size=128)
    n_llama, n_gpt = TinyLlama(c).num_parameters(), GPT(c).num_parameters()
    print(f"    TinyLlama {n_llama:,} vs TinyGPT {n_gpt:,} parameters")
    assert abs(n_llama - n_gpt) / n_gpt < 0.05


def test_runs_beyond_training_length():
    """No position table, so it can process sequences longer than block_size (quality aside)."""
    model = TinyLlama(cfg(block_size=16)).eval()
    logits, _ = model(torch.randint(0, 40, (1, 40)))
    assert logits.shape == (1, 40, 40)


def test_works_with_bonus13_sampling():
    model = TinyLlama(cfg())
    out = generate(model, torch.randint(0, 40, (2, 3)), 10, temperature=0.8, top_k=5)
    assert out.shape == (2, 13)


def test_learns_a_pattern():
    model = TinyLlama(cfg(vocab_size=8))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    data = torch.arange(8).repeat(40)
    for _ in range(150):
        ix = torch.randint(0, len(data) - 17, (16,))
        x = torch.stack([data[i:i + 16] for i in ix])
        y = torch.stack([data[i + 1:i + 17] for i in ix])
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward(); opt.step()
    out = generate(model, torch.tensor([[2, 3, 4]]), 10, temperature=0)
    assert out[0].tolist() == [2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  ✓ {name}")
    print("Bonus Step 17: all tests passed")
