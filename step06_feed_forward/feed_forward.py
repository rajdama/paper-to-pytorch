"""
Step 06 - Position-wise Feed-Forward Network
Paper: Section 3.3

    FFN(x) = max(0, x W1 + b1) W2 + b2

The same small MLP is applied to every position independently.
Attention mixes information BETWEEN tokens; the FFN processes each token ON ITS OWN.
"""

import torch
import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)    # expand: 512 -> 2048
        self.linear2 = nn.Linear(d_ff, d_model)    # project back: 2048 -> 512
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model) -> same shape"""
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))
