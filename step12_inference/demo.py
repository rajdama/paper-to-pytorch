"""
Use the model trained in Step 11 to reverse sequences.
First train it:   python -m step11_training.train_reverse
Then run:         python -m step12_inference.demo
Or your own input: python -m step12_inference.demo 3 1 4 1 5 9 2 6
"""

import sys
from pathlib import Path

import torch

from step10_full_transformer.transformer import Transformer
from step11_training.training import BOS, EOS, FIRST_SYMBOL, PAD, reverse_task_batch
from step12_inference.decoding import beam_search, greedy_decode

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoints" / "reverse.pt"


def to_ids(digits):        # digit d -> token id d + FIRST_SYMBOL
    return [d + FIRST_SYMBOL for d in digits]


def to_digits(ids):
    return [i - FIRST_SYMBOL for i in ids if i >= FIRST_SYMBOL]


def main():
    if not CKPT.exists():
        sys.exit("No trained model found. Run:  python -m step11_training.train_reverse")
    ckpt = torch.load(CKPT, map_location="cpu")
    model = Transformer(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    if len(sys.argv) > 1:                                  # user-provided sequence
        digits = [int(d) for d in sys.argv[1:]]
        assert all(0 <= d <= 9 for d in digits), "use digits 0-9"
        src = torch.tensor([to_ids(digits) + [EOS]])
        g = greedy_decode(model, src, len(digits) + 2, BOS, EOS)
        b = beam_search(model, src, len(digits) + 2, BOS, EOS, beam_size=4)
        print(f"input : {digits}")
        print(f"greedy: {to_digits(g[0].tolist())}")
        print(f"beam  : {to_digits(b[0].tolist())}")
        return

    torch.manual_seed(123)
    src, tgt = reverse_task_batch(200)
    max_len = tgt.size(1)
    expected = [to_digits(t.tolist()) for t in tgt]

    greedy = greedy_decode(model, src, max_len, BOS, EOS)
    g_correct = sum(to_digits(g.tolist()) == e for g, e in zip(greedy, expected))

    b_correct = 0
    for i in range(len(src)):
        row = src[i:i + 1, : int((src[i] != PAD).sum())]        # strip padding for single-sentence beam search
        b = beam_search(model, row, max_len, BOS, EOS, beam_size=4)
        b_correct += to_digits(b[0].tolist()) == expected[i]

    print("examples:")
    for i in range(5):
        print(f"  {to_digits(src[i].tolist())}  ->  {to_digits(greedy[i].tolist())}")
    print(f"\nexact-match accuracy on 200 unseen sequences")
    print(f"  greedy decoding:      {g_correct / 2:.1f}%")
    print(f"  beam search (k = 4):  {b_correct / 2:.1f}%")


if __name__ == "__main__":
    main()
