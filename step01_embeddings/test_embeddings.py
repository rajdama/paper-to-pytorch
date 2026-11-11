"""Run from the repo root:  python -m step01_embeddings.test_embeddings   (or: pytest)"""

import math
import torch

from step01_embeddings.embeddings import TokenEmbedding


def test_output_shape():
    emb = TokenEmbedding(vocab_size=100, d_model=32)
    ids = torch.randint(0, 100, (4, 10))          # 4 sequences, 10 tokens each
    assert emb(ids).shape == (4, 10, 32)


def test_scaled_by_sqrt_d_model():
    emb = TokenEmbedding(vocab_size=100, d_model=64)
    ids = torch.tensor([[5]])
    raw = emb.embedding.weight[5]
    assert torch.allclose(emb(ids)[0, 0], raw * math.sqrt(64))


def test_same_token_same_vector():
    emb = TokenEmbedding(vocab_size=100, d_model=16)
    out = emb(torch.tensor([[7, 3, 7]]))
    assert torch.equal(out[0, 0], out[0, 2])       # token 7 twice -> identical vectors
    assert not torch.equal(out[0, 0], out[0, 1])


def test_padding_token_is_zero():
    emb = TokenEmbedding(vocab_size=100, d_model=16, padding_idx=0)
    out = emb(torch.tensor([[0, 1]]))
    assert torch.all(out[0, 0] == 0)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  ✓ {name}")
    print("Step 01: all tests passed")
