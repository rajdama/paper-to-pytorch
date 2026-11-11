"""
Step 01 - Token Embeddings
Paper: Section 3.4 "Embeddings and Softmax"

Turns token ids (integers) into vectors of size d_model.
The paper multiplies the embedding by sqrt(d_model) so its scale matches the
positional encoding that gets added next (Step 02).
"""

import math
import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, padding_idx: int | None = None):
        super().__init__()
        self.d_model = d_model
        # A lookup table: row i is the learned vector for token i
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: (batch, seq_len) integer tensor
        returns:   (batch, seq_len, d_model)
        """
        return self.embedding(token_ids) * math.sqrt(self.d_model)
