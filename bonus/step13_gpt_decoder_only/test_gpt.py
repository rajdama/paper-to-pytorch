"""Run from the repo root:  python -m bonus.step13_gpt_decoder_only.test_gpt   (or: pytest)"""

import math

import torch

from bonus.step13_gpt_decoder_only.char_tokenizer import CharTokenizer
from bonus.step13_gpt_decoder_only.gpt import GPT, GPTConfig
from bonus.step13_gpt_decoder_only.sampling import filter_logits, generate, sample_next

torch.manual_seed(0)


def tiny(vocab=20, **kw):
    cfg = dict(vocab_size=vocab, block_size=16, d_model=32, num_heads=4, num_layers=2, dropout=0.0)
    cfg.update(kw)
    return GPT(GPTConfig(**cfg))


def test_logits_shape_and_loss():
    model = tiny()
    idx = torch.randint(0, 20, (3, 10))
    logits, loss = model(idx, idx)
    assert logits.shape == (3, 10, 20)
    assert loss.shape == () and loss.item() > 0


def test_initial_loss_is_about_log_vocab():
    """Untrained, the model should be roughly uniform: loss ~ ln(vocab_size)."""
    model = tiny(vocab=50).eval()
    idx = torch.randint(0, 50, (8, 16))
    _, loss = model(idx, torch.randint(0, 50, (8, 16)))
    assert abs(loss.item() - math.log(50)) < 0.3


def test_is_causal():
    """The prediction at position t must not depend on tokens after t."""
    model = tiny().eval()
    idx = torch.randint(0, 20, (1, 12))
    logits1, _ = model(idx)
    idx2 = idx.clone()
    idx2[0, 8:] = (idx2[0, 8:] + 1) % 20
    logits2, _ = model(idx2)
    assert torch.allclose(logits1[:, :8], logits2[:, :8], atol=1e-5)
    assert not torch.allclose(logits1[:, 8:], logits2[:, 8:])


def test_no_cross_attention_and_weight_tying():
    model = tiny()
    assert not any(hasattr(b, "cross_attn") for b in model.blocks)
    assert model.lm_head.weight is model.tok_emb.weight


def test_context_longer_than_block_size_is_rejected():
    model = tiny(block_size=8)
    try:
        model(torch.zeros(1, 9, dtype=torch.long))
    except AssertionError:
        return
    raise AssertionError("expected an error for a too-long sequence")


def test_top_k_keeps_k_tokens():
    logits = torch.randn(2, 30)
    out = filter_logits(logits, top_k=5)
    assert torch.all(torch.isfinite(out).sum(-1) == 5)
    assert torch.equal(out.argmax(-1), logits.argmax(-1))


def test_top_p_keeps_the_nucleus():
    probs = torch.tensor([[0.05, 0.5, 0.15, 0.3]])     # sorted: 0.5, 0.3, 0.15, 0.05
    logits = probs.log()
    keep = lambda p: torch.isfinite(filter_logits(logits, top_p=p))[0].tolist()
    assert keep(0.4) == [False, True, False, False]   # 0.5 alone covers 0.4
    assert keep(0.7) == [False, True, False, True]    # 0.5 + 0.3 = 0.8 covers 0.7
    assert keep(0.9) == [False, True, True, True]     # 0.5 + 0.3 + 0.15 = 0.95
    assert keep(1.0) == [True, True, True, True]


def test_temperature_zero_is_greedy():
    logits = torch.randn(4, 30)
    assert torch.equal(sample_next(logits, temperature=0).squeeze(1), logits.argmax(-1))


def test_generate_length_and_prompt_kept():
    model = tiny(block_size=8)
    prompt = torch.randint(0, 20, (2, 3))
    out = generate(model, prompt, max_new_tokens=12)      # longer than block_size: context is cropped
    assert out.shape == (2, 15)
    assert torch.equal(out[:, :3], prompt)


def test_char_tokenizer_roundtrip():
    tok = CharTokenizer("hello world")
    assert tok.vocab_size == 8
    assert tok.decode(tok.encode("hello")) == "hello"
    assert CharTokenizer.from_state(tok.state()).encode("low") == tok.encode("low")


def test_learns_to_continue_a_pattern():
    """Overfit a repeating string: greedy generation should then continue it exactly."""
    text = "abcdefgh" * 40
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text))
    model = tiny(vocab=tok.vocab_size, block_size=16)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(150):
        ix = torch.randint(0, len(data) - 17, (16,))
        x = torch.stack([data[i:i + 16] for i in ix])
        y = torch.stack([data[i + 1:i + 17] for i in ix])
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward(); opt.step()
    out = generate(model, torch.tensor([tok.encode("cde")]), 12, temperature=0)
    assert tok.decode(out[0].tolist()) == "cdefghabcdefgha"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  ✓ {name}")
    print("Bonus Step 13: all tests passed")
