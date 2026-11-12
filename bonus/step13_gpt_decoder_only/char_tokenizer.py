"""
The simplest tokenizer: one token per character.
Bonus Step 14 replaces it with Byte Pair Encoding.
"""


class CharTokenizer:
    def __init__(self, text: str):
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for c, i in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    def state(self) -> dict:
        return {"type": "char", "chars": "".join(self.chars)}

    @classmethod
    def from_state(cls, state: dict) -> "CharTokenizer":
        return cls(state["chars"])
