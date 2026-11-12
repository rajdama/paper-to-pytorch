"""
Bonus Step 16 - LoRA: Low-Rank Adaptation
Paper: Hu et al., 2021, "LoRA: Low-Rank Adaptation of Large Language Models"

Fine-tuning changes a weight matrix W (d_out x d_in) by some update dW. LoRA's bet:
that update has LOW RANK, so write it as the product of two thin matrices

    W' = W + (alpha / r) * B @ A          A: (r x d_in),  B: (d_out x r),  r << d

Freeze W, train only A and B. With d = 512 and r = 8 that's 8,192 trainable numbers
instead of 262,144 per matrix (3%).

B starts at ZERO, so at the start of fine-tuning the model is exactly the pretrained one.
After training, B @ A can be MERGED into W: no extra cost at inference time.
"""

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)                       # the pretrained weights stay frozen
        self.rank, self.scale = rank, alpha / rank
        self.A = nn.Parameter(torch.empty(rank, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))   # zero: no change at start
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))  # same init nn.Linear uses

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale

    def delta_weight(self) -> torch.Tensor:
        return self.B @ self.A * self.scale

    def merged(self) -> nn.Linear:
        """A plain nn.Linear with the update folded in: W + scale * B A."""
        lin = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None)
        with torch.no_grad():
            lin.weight.copy_(self.base.weight + self.delta_weight())
            if self.base.bias is not None:
                lin.bias.copy_(self.base.bias)
        return lin


def apply_lora(model: nn.Module, targets=("w_q", "w_v"), rank: int = 8, alpha: float = 16.0):
    """
    Freeze every parameter of `model`, then wrap the Linear layers whose attribute name is
    in `targets` with LoRALinear. The LoRA paper's default adapts the query and value
    projections of every attention layer (Section 7.1).
    Returns the list of wrapped modules.
    """
    for p in model.parameters():
        p.requires_grad_(False)
    wrapped = []
    for module in list(model.modules()):
        for name in targets:
            child = getattr(module, name, None)
            if isinstance(child, nn.Linear):
                lora = LoRALinear(child, rank, alpha)
                setattr(module, name, lora)
                wrapped.append(lora)
    return wrapped


def merge_lora(model: nn.Module) -> nn.Module:
    """Replace every LoRALinear with its merged nn.Linear (in place)."""
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                setattr(module, name, child.merged())
    return model


def count_trainable(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
