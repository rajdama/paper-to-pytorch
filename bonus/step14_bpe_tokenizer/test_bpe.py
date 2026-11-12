"""Run from the repo root:  python -m bonus.step14_bpe_tokenizer.test_bpe   (pure Python, no torch needed)"""

import re
import tempfile
from pathlib import Path

from bonus.step14_bpe_tokenizer.bpe import SPLIT_PATTERN, BPETokenizer, get_pair_counts, merge

SAMPLE = ("To be, or not to be, that is the question:\n"
          "Whether 'tis nobler in the mind to suffer\n"
          "The slings and arrows of outrageous fortune,\n"
          "Or to take arms against a sea of troubles\n") * 5
UNICODE = "héllo wörld! naïve café — 日本語のテキスト 🤖🚀 snake_case 3.14 ²√ ünïcödé\t\n  end"


def test_get_pair_counts():
    assert get_pair_counts([1, 2, 3, 1, 2]) == {(1, 2): 2, (2, 3): 1, (3, 1): 1}


def test_merge():
    assert merge([1, 2, 3, 1, 2], (1, 2), 9) == [9, 3, 9]
    assert merge([1, 1, 1], (1, 1), 9) == [9, 1]              # non-overlapping, left to right


def test_wikipedia_example():
    """'aaabdaaabac' with 3 merges (basic mode, no pre-splitting) becomes 5 tokens."""
    tok = BPETokenizer(pattern=None)
    tok.train("aaabdaaabac", vocab_size=256 + 3)
    ids = tok.encode("aaabdaaabac")
    assert len(ids) == 5
    assert list(tok.merges)[0] == (ord("a"), ord("a"))          # 'aa' is the most frequent pair
    assert tok.decode(ids) == "aaabdaaabac"


def test_roundtrip_any_text():
    tok = BPETokenizer()
    tok.train(SAMPLE, vocab_size=300)
    for text in (SAMPLE, UNICODE, "", " ", "\n\n\n", "zzzz never seen before"):
        assert tok.decode(tok.encode(text)) == text


def test_split_pattern_never_drops_characters():
    for text in (UNICODE, SAMPLE, "a_b__c", "tab\there", "x́y"):   # ́ = combining accent
        assert "".join(re.findall(SPLIT_PATTERN, text)) == text


def test_merges_compress_text():
    tok = BPETokenizer()
    tok.train(SAMPLE, vocab_size=320)
    assert len(tok.encode(SAMPLE)) < 0.5 * len(SAMPLE.encode("utf-8"))


def test_merges_do_not_cross_word_boundaries():
    tok = BPETokenizer()
    tok.train(SAMPLE, vocab_size=330)
    for token in tok.token_strings(SAMPLE):
        stripped = token.strip(" ")
        assert " " not in stripped                                  # a space can only lead a token


def test_vocab_bytes_are_concatenations():
    tok = BPETokenizer()
    tok.train(SAMPLE, vocab_size=290)
    for (a, b), new in tok.merges.items():
        assert tok.vocab[new] == tok.vocab[a] + tok.vocab[b]
        assert a < new and b < new                                  # built only from older tokens


def test_save_and_load():
    tok = BPETokenizer()
    tok.train(SAMPLE, vocab_size=300)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "tok.json"
        tok.save(path)
        loaded = BPETokenizer.load(path)
    assert loaded.encode(UNICODE + SAMPLE) == tok.encode(UNICODE + SAMPLE)
    assert loaded.vocab_size == tok.vocab_size == 300


def test_training_is_deterministic():
    a, b = BPETokenizer(), BPETokenizer()
    a.train(SAMPLE, 300)
    b.train(SAMPLE, 300)
    assert a.merges == b.merges


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  ✓ {name}")
    print("Bonus Step 14: all tests passed")
