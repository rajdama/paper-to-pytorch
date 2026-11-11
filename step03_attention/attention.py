"""
Step 03 - Scaled Dot-Product Attention
Paper: Section 3.2.1 "Scaled Dot-Product Attention" (and Figure 2, left)

    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

This single function lives inside EVERY attention box in the architecture.
"""

import math
import torch
import torch.nn.functional as F


def scaled_dot_product_attention(q: torch.Tensor,
                                 k: torch.Tensor,
                                 v: torch.Tensor,
                                 mask: torch.Tensor | None = None):
    """
    q: (..., seq_q, d_k)   - what each token is looking for
    k: (..., seq_k, d_k)   - what each token offers
    v: (..., seq_k, d_v)   - the information each token passes on
    mask: broadcastable to (..., seq_q, seq_k); 1/True = may attend, 0/False = blocked
          (we build real masks in Step 04)

    returns:
        output:  (..., seq_q, d_v)
        weights: (..., seq_q, seq_k)  - each row sums to 1
    The "..." lets the same code handle (batch,) now and (batch, heads) in Step 05.
    """
    d_k = q.size(-1)

    # 1. Similarity of every query with every key
    scores = q @ k.transpose(-2, -1)                 # (..., seq_q, seq_k)

    # 2. Scale - without this, large d_k pushes softmax into tiny-gradient regions
    scores = scores / math.sqrt(d_k)

    # 3. Optional mask: blocked positions get -inf, so softmax gives them 0
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    # 4. Scores -> probabilities over the keys
    weights = F.softmax(scores, dim=-1)

    # 5. Weighted average of the values
    output = weights @ v                             # (..., seq_q, d_v)
    return output, weights
