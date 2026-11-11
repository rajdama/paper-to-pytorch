"""Run from the repo root:  python -m step12_inference.test_decoding"""

import torch

from step10_full_transformer.transformer import Transformer
from step11_training.training import BOS, EOS, PAD
from step12_inference.decoding import beam_search, greedy_decode

torch.manual_seed(0)
V = 13


def tiny():
    return Transformer(V, V, d_model=32, num_heads=4, num_layers=2, d_ff=64,
                       dropout=0.0, share_src_tgt_embeddings=True)


def test_greedy_shape_and_start():
    model = tiny()
    src = torch.randint(3, V, (4, 7))
    out = greedy_decode(model, src, max_len=10, bos_idx=BOS, eos_idx=EOS)
    assert out.shape[0] == 4 and out.shape[1] <= 10
    assert torch.all(out[:, 0] == BOS)


def test_greedy_stops_at_eos():
    model = tiny()
    with torch.no_grad():
        model.generator.bias[EOS] = 1e4          # rig the model to always say <eos>
    out = greedy_decode(model, torch.randint(3, V, (2, 5)), 10, BOS, EOS)
    assert out.tolist() == [[BOS, EOS], [BOS, EOS]]


def test_pad_after_eos():
    model = tiny()
    out = greedy_decode(model, torch.randint(3, V, (8, 6)), 12, BOS, EOS)
    for row in out.tolist():
        if EOS in row:
            assert all(t == PAD for t in row[row.index(EOS) + 1:])


def test_beam_size_one_equals_greedy():
    model = tiny()
    for _ in range(5):
        src = torch.randint(3, V, (1, 6))
        g = greedy_decode(model, src, 10, BOS, EOS)
        b = beam_search(model, src, 10, BOS, EOS, beam_size=1)
        assert g.tolist() == b.tolist()


def test_beam_search_stops_at_eos():
    model = tiny()
    with torch.no_grad():
        model.generator.bias[EOS] = 1e4
    out = beam_search(model, torch.randint(3, V, (1, 5)), 10, BOS, EOS, beam_size=4)
    assert out.tolist() == [[BOS, EOS]]


def test_beam_search_output_format():
    model = tiny()
    out = beam_search(model, torch.randint(3, V, (1, 6)), 9, BOS, EOS, beam_size=3)
    assert out.shape[0] == 1 and out.shape[1] <= 9 and out[0, 0] == BOS


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  ✓ {name}")
    print("Step 12: all tests passed")
