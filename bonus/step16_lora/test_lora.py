"""Run from the repo root:  python -m bonus.step16_lora.test_lora"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from bonus.step13_gpt_decoder_only.gpt import GPT, GPTConfig
from bonus.step16_lora.lora import LoRALinear, apply_lora, count_trainable, merge_lora

torch.manual_seed(0)


def tiny_gpt():
    return GPT(GPTConfig(vocab_size=30, block_size=16, d_model=32, num_heads=4, num_layers=2, dropout=0.0))


def test_starts_identical_to_base():
    base = nn.Linear(16, 8)
    lora = LoRALinear(base, rank=4)
    x = torch.randn(3, 16)
    assert torch.allclose(lora(x), base(x))                # B = 0, so no change yet


def test_only_lora_parameters_train():
    base = nn.Linear(16, 8)
    lora = LoRALinear(base, rank=4)
    lora(torch.randn(5, 16)).sum().backward()
    assert base.weight.grad is None and base.bias.grad is None
    assert lora.A.grad is not None and lora.B.grad is not None


def test_merge_gives_the_same_function():
    lora = LoRALinear(nn.Linear(16, 8), rank=4)
    with torch.no_grad():
        lora.B.normal_()                                   # pretend it was trained
    x = torch.randn(6, 16)
    assert torch.allclose(lora(x), lora.merged()(x), atol=1e-5)


def test_update_has_low_rank():
    lora = LoRALinear(nn.Linear(64, 64), rank=4)
    with torch.no_grad():
        lora.B.normal_()
    assert torch.linalg.matrix_rank(lora.delta_weight()) == 4


def test_apply_to_gpt():
    model = tiny_gpt()
    wrapped = apply_lora(model, targets=("w_q", "w_v"), rank=4)
    assert len(wrapped) == 2 * 2                            # 2 layers x (query, value)
    trainable, total = count_trainable(model)
    assert trainable == 4 * (4 * 32 + 32 * 4)               # each: A (4x32) + B (32x4)
    assert trainable < 0.05 * total


def test_gpt_unchanged_before_training_and_after_merge():
    model = tiny_gpt().eval()
    idx = torch.randint(0, 30, (2, 10))
    before, _ = model(idx)
    apply_lora(model, rank=4)
    after_wrap, _ = model(idx)
    assert torch.allclose(before, after_wrap, atol=1e-6)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, LoRALinear):
                m.B.normal_(0, 0.1)
    adapted, _ = model(idx)
    merge_lora(model)
    assert not any(isinstance(m, LoRALinear) for m in model.modules())
    merged, _ = model(idx)
    assert torch.allclose(adapted, merged, atol=1e-5)


def test_lora_can_fit_a_low_rank_change():
    """If the 'true' change to a frozen layer is rank 2, a rank-2 LoRA can learn it."""
    base = nn.Linear(20, 20)
    target = nn.Linear(20, 20)
    with torch.no_grad():
        target.weight.copy_(base.weight + torch.randn(20, 2) @ torch.randn(2, 20) * 0.5)
        target.bias.copy_(base.bias)
    lora = LoRALinear(base, rank=2, alpha=2)
    opt = torch.optim.Adam([lora.A, lora.B], lr=1e-2)
    x_test = torch.randn(256, 20)
    with torch.no_grad():
        start = F.mse_loss(lora(x_test), target(x_test)).item()
    for _ in range(1500):
        x = torch.randn(64, 20)
        loss = F.mse_loss(lora(x), target(x))
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        end = F.mse_loss(lora(x_test), target(x_test)).item()
    assert end < 0.01 * start


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  ✓ {name}")
    print("Bonus Step 16: all tests passed")
