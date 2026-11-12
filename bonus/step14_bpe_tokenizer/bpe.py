"""
Bonus Step 14 - Byte Pair Encoding (BPE) tokenizer, from scratch

The tokenizer family used by GPT-2, GPT-4, LLaMA 3 and most modern LLMs.

Training:
    1. start from the 256 possible bytes (so ANY text can be encoded, no <unk> needed)
    2. count every adjacent pair of tokens in the training text
    3. merge the most frequent pair into a new token
    4. repeat until the vocabulary is as big as you want

Encoding replays the learned merges in the order they were learned.

Like GPT-2, the text is first split into chunks (words, numbers, punctuation, spaces)
so merges never cross a word boundary: "dog." and "dog!" share the token " dog".
"""

import json
import re
from collections import Counter

# GPT-2-style split, written for Python's built-in `re`:
#   contractions | optional space + letters | optional space + digits |
#   optional space + punctuation/symbols | whitespace runs
SPLIT_PATTERN = r"'(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+"


def get_pair_counts(ids, counts=None, weight=1):
    """Count adjacent pairs: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}"""
    counts = Counter() if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] += weight
    return counts


def merge(ids, pair, new_id):
    """Replace every occurrence of `pair` with `new_id`: ([1, 2, 3, 1, 2], (1, 2), 9) -> [9, 3, 9]"""
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self, pattern: str | None = SPLIT_PATTERN):
        """pattern=None trains on the raw byte stream with no pre-splitting (the 'basic' version)."""
        self.pattern = pattern
        self.merges: dict[tuple[int, int], int] = {}            # (a, b) -> new token id
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self._cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges)

    def _split(self, text: str) -> list[str]:
        if self.pattern is None:
            return [text]
        chunks = re.findall(self.pattern, text)
        assert "".join(chunks) == text, "split pattern dropped characters"
        return chunks

    # ------------------------------------------------------------------ training
    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        assert vocab_size >= 256, "the 256 byte tokens are always included"
        # Work on UNIQUE chunks with their counts: "the" appearing 20,000 times
        # is stored once with weight 20,000. This is what makes training fast.
        chunk_counts = Counter(self._split(text))
        words = [[list(chunk.encode("utf-8")), n] for chunk, n in chunk_counts.items()]

        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self._cache = {}
        for i in range(vocab_size - 256):
            stats = Counter()
            for ids, n in words:
                get_pair_counts(ids, stats, n)
            if not stats:
                break                                           # nothing left to merge
            pair = max(stats, key=stats.get)                    # the most frequent pair
            new_id = 256 + i
            for w in words:
                if len(w[0]) > 1:
                    w[0] = merge(w[0], pair, new_id)
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose:
                print(f"merge {i + 1:4d}: {self.vocab[pair[0]]!r} + {self.vocab[pair[1]]!r} "
                      f"-> {self.vocab[new_id]!r}  ({stats[pair]:,} times)")

    # ------------------------------------------------------------------ encode / decode
    def _encode_chunk(self, chunk: str) -> list[int]:
        if chunk in self._cache:
            return self._cache[chunk]
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            # apply the EARLIEST-learned merge available, exactly as in training
            pair = min(set(zip(ids, ids[1:])), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        self._cache[chunk] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        out = []
        for chunk in self._split(text):
            out.extend(self._encode_chunk(chunk))
        return out

    def decode(self, ids) -> str:
        data = b"".join(self.vocab[int(i)] for i in ids)
        # a slice of tokens may end in the middle of a multi-byte UTF-8 character
        return data.decode("utf-8", errors="replace")

    def token_strings(self, text: str) -> list[str]:
        """How the text is split into tokens, for visualising."""
        return [self.vocab[i].decode("utf-8", errors="replace") for i in self.encode(text)]

    # ------------------------------------------------------------------ save / load
    def state(self) -> dict:
        return {"type": "bpe", "pattern": self.pattern,
                "merges": [[a, b, new] for (a, b), new in self.merges.items()]}

    @classmethod
    def from_state(cls, state: dict) -> "BPETokenizer":
        tok = cls(state["pattern"])
        for a, b, new in state["merges"]:               # stored in learning order
            tok.merges[(a, b)] = new
            tok.vocab[new] = tok.vocab[a] + tok.vocab[b]
        return tok

    def save(self, path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.state(), f)

    @classmethod
    def load(cls, path) -> "BPETokenizer":
        with open(path, encoding="utf-8") as f:
            return cls.from_state(json.load(f))
