"""Run from the repo root:  python -m bonus.step15_kv_cache.test_kv_cache"""

import torch

from bonus.step13_gpt_decoder_only.gpt import GPT, GPTConfig
from bonus.step13_gpt_decoder_only.sampling import generate
from bonus.step15_kv_cache.kv_cache import KVCache, forward_with_cache, generate_with_kv_cache

torch.manual_seed(0)


def tiny(block_size=32):
    cfg = GPTConfig(vocab_size=30, block_size=block_size, d_model=32, num_heads=4,
                    num_layers=3, dropout=0.1)                    # dropout must be off in eval
    return GPT(cfg).eval()


def test_prefill_matches_full_forward():
    model = tiny()
    idx = torch.randint(0, 30, (2, 10))
    full, _ = model(idx)
    cached = forward_with_cache(model, idx, KVCache(3))
    assert torch.allclose(full, cached, atol=1e-5)


def test_token_by_token_matches_full_forward():
    """Feeding one token at a time through the cache gives the same logits at every position."""
    model = tiny()
    idx = torch.randint(0, 30, (2, 12))
    full, _ = model(idx)
    cache = KVCache(3)
    for t in range(12):
        step_logits = forward_with_cache(model, idx[:, t:t + 1], cache)
        assert torch.allclose(step_logits[:, 0], full[:, t], atol=1e-5), f"mismatch at position {t}"


def test_prefill_then_decode_matches():
    model = tiny()
    idx = torch.randint(0, 30, (1, 15))
    full, _ = model(idx)
    cache = KVCache(3)
    forward_with_cache(model, idx[:, :9], cache)                 # prompt
    rest = forward_with_cache(model, idx[:, 9:], cache)          # several new tokens at once
    assert torch.allclose(rest, full[:, 9:], atol=1e-5)


def test_greedy_generation_identical_with_and_without_cache():
    model = tiny()
    prompt = torch.randint(0, 30, (3, 5))
    slow = generate(model, prompt, 20, temperature=0)
    fast = generate_with_kv_cache(model, prompt, 20, temperature=0)
    assert torch.equal(slow, fast)


def test_sampled_generation_identical_with_same_seed():
    model = tiny()
    prompt = torch.randint(0, 30, (2, 4))
    slow = generate(model, prompt, 15, temperature=1.0, top_k=10,
                    generator=torch.Generator().manual_seed(7))
    fast = generate_with_kv_cache(model, prompt, 15, temperature=1.0, top_k=10,
                                  generator=torch.Generator().manual_seed(7))
    assert torch.equal(slow, fast)


def test_cache_grows_by_one_token_per_step():
    model = tiny()
    cache = KVCache(3)
    forward_with_cache(model, torch.randint(0, 30, (1, 6)), cache)
    assert cache.length == 6
    for i in range(4):
        forward_with_cache(model, torch.randint(0, 30, (1, 1)), cache)
        assert cache.length == 7 + i
    # 3 layers x (K and V) x 1 batch x 4 heads x 10 tokens x 8 dims x 4 bytes
    assert cache.num_bytes() == 3 * 2 * 1 * 4 * 10 * 8 * 4


def test_cannot_go_past_block_size():
    model = tiny(block_size=8)
    cache = KVCache(3)
    forward_with_cache(model, torch.randint(0, 30, (1, 8)), cache)
    try:
        forward_with_cache(model, torch.randint(0, 30, (1, 1)), cache)
    except AssertionError:
        return
    raise AssertionError("expected an error beyond block_size")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  ✓ {name}")
    print("Bonus Step 15: all tests passed")
