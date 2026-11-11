"""Run from the repo root:  python -m step11_training.test_training"""

import torch
import torch.nn.functional as F

from step10_full_transformer.transformer import Transformer
from step11_training.training import (BOS, EOS, PAD, LabelSmoothingLoss, make_optimizer,
                                      noam_lr, reverse_task_batch)

torch.manual_seed(0)


def test_label_smoothing_matches_pytorch():
    logits = torch.randn(20, 13)
    target = torch.randint(0, 13, (20,))
    target[:4] = PAD                                    # some padding to ignore
    ours = LabelSmoothingLoss(0.1, ignore_index=PAD)(logits, target)
    ref = F.cross_entropy(logits, target, ignore_index=PAD, label_smoothing=0.1)
    assert torch.allclose(ours, ref, atol=1e-6)


def test_zero_smoothing_is_plain_cross_entropy():
    logits, target = torch.randn(10, 7), torch.randint(1, 7, (10,))
    assert torch.allclose(LabelSmoothingLoss(0.0)(logits, target), F.cross_entropy(logits, target))


def test_noam_schedule_shape():
    d, w = 512, 4000
    lrs = [noam_lr(s, d, w) for s in range(1, 20001)]
    peak = max(range(len(lrs)), key=lrs.__getitem__) + 1
    assert peak == w                                     # rises until warmup, then decays
    assert abs(lrs[w - 1] - d ** -0.5 * w ** -0.5) < 1e-12
    assert lrs[0] < lrs[999] < lrs[w - 1] > lrs[9999] > lrs[19999]


def test_optimizer_follows_schedule():
    model = torch.nn.Linear(4, 4)
    opt, sched = make_optimizer(model, d_model=512, warmup=100)
    for step in range(1, 6):
        assert abs(opt.param_groups[0]["lr"] - noam_lr(step, 512, 100)) < 1e-12
        opt.step(); sched.step()


def test_reverse_batch_format():
    src, tgt = reverse_task_batch(16)
    for s, t in zip(src, tgt):
        n = (s == EOS).nonzero()[0].item()               # length before EOS
        assert t[0] == BOS and t[n + 1] == EOS
        assert torch.equal(t[1:n + 1], s[:n].flip(0))    # target = reversed source
        assert torch.all(s[n + 1:] == PAD) and torch.all(t[n + 2:] == PAD)


def test_short_training_run_learns():
    """300 steps on the reverse task: loss must drop clearly."""
    model = Transformer(13, 13, d_model=32, num_heads=4, num_layers=2, d_ff=64,
                        dropout=0.0, share_src_tgt_embeddings=True)
    crit = LabelSmoothingLoss(0.1)
    opt, sched = make_optimizer(model, 32, warmup=100, factor=0.5)
    losses = []
    for _ in range(300):
        src, tgt = reverse_task_batch(32, max_len=6)
        logits = model(src, tgt[:, :-1])
        loss = crit(logits.reshape(-1, 13), tgt[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        losses.append(loss.item())
    first, last = sum(losses[:20]) / 20, sum(losses[-20:]) / 20
    print(f"    loss {first:.2f} -> {last:.2f}")
    assert last < 0.75 * first


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  ✓ {name}")
    print("Step 11: all tests passed")
