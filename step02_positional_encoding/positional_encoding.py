"""
Step 02 - Sinusoidal Positional Encoding
Paper: Section 3.5 "Positional Encoding"

Attention has no sense of order: shuffle the tokens and it gives the same answer.
So we add a unique "position signature" to every token embedding:

    PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
"""

import math
import torch
import torch.nn as nn


def sinusoidal_table(max_len: int, d_model: int) -> torch.Tensor:
    """Returns the (max_len, d_model) table of positional encodings."""
    position = torch.arange(max_len).unsqueeze(1).float()             # (max_len, 1)
    # 1 / 10000^(2i/d_model), computed in log space for numerical stability
    div_term = torch.exp(torch.arange(0, d_model, 2).float()
                         * (-math.log(10000.0) / d_model))           # (d_model/2,)
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)                     # even dims
    pe[:, 1::2] = torch.cos(position * div_term)                     # odd dims
    return pe


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even (sin/cos pairs)"
        self.dropout = nn.Dropout(dropout)
        # A buffer is saved with the model and moved to GPU with it, but is NOT trained
        self.register_buffer("pe", sinusoidal_table(max_len, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model) - the token embeddings from Step 01
        returns the same shape, with position information added
        """
        seq_len = x.size(1)
        return self.dropout(x + self.pe[:seq_len])
