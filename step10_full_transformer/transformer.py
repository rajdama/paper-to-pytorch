"""
Step 10 - The full Transformer
Paper: Section 3 and Figure 1

Every box in the diagram, wired together:
  src tokens -> Embedding (01) + Positional Encoding (02) -> Encoder (08) -> memory
  tgt tokens -> Embedding (01) + Positional Encoding (02) -> Decoder (09, reads memory)
             -> Linear -> (softmax) -> probabilities over the target vocabulary
"""

import torch
import torch.nn as nn

from step01_embeddings.embeddings import TokenEmbedding
from step02_positional_encoding.positional_encoding import PositionalEncoding
from step04_masking.masks import make_decoder_mask, make_padding_mask
from step08_encoder.encoder import Encoder
from step09_decoder.decoder import Decoder


class Transformer(nn.Module):
    def __init__(self,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 d_model: int = 512,        # paper "base" model defaults
                 num_heads: int = 8,
                 num_layers: int = 6,
                 d_ff: int = 2048,
                 dropout: float = 0.1,
                 max_len: int = 5000,
                 pad_idx: int = 0,
                 share_src_tgt_embeddings: bool = False,
                 pre_norm: bool = False):
        super().__init__()
        self.pad_idx = pad_idx

        self.src_embed = TokenEmbedding(src_vocab_size, d_model, pad_idx)
        if share_src_tgt_embeddings:
            # the paper uses one shared BPE vocabulary for both languages
            assert src_vocab_size == tgt_vocab_size, "sharing needs one common vocabulary"
            self.tgt_embed = self.src_embed
        else:
            self.tgt_embed = TokenEmbedding(tgt_vocab_size, d_model, pad_idx)

        self.positional = PositionalEncoding(d_model, max_len, dropout)
        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, dropout, pre_norm)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff, dropout, pre_norm)
        self.generator = nn.Linear(d_model, tgt_vocab_size)   # the "Linear" box

        self._init_weights()
        # Section 3.4: the pre-softmax Linear shares its weight matrix with the target embedding
        self.generator.weight = self.tgt_embed.embedding.weight

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        with torch.no_grad():                                 # keep <pad> embeddings at zero
            self.src_embed.embedding.weight[self.pad_idx].zero_()
            self.tgt_embed.embedding.weight[self.pad_idx].zero_()

    # --- the two halves, also used separately at inference time (Step 12) ---
    def encode(self, src: torch.Tensor):
        src_mask = make_padding_mask(src, self.pad_idx)
        memory = self.encoder(self.positional(self.src_embed(src)), src_mask)
        return memory, src_mask

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, src_mask: torch.Tensor):
        tgt_mask = make_decoder_mask(tgt, self.pad_idx)
        out = self.decoder(self.positional(self.tgt_embed(tgt)), memory, src_mask, tgt_mask)
        return self.generator(out)          # logits; softmax is applied in the loss / sampler

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        src: (B, src_len) source token ids
        tgt: (B, tgt_len) target token ids, shifted right (starts with <bos>)
        returns logits: (B, tgt_len, tgt_vocab_size)
        """
        memory, src_mask = self.encode(src)
        return self.decode(tgt, memory, src_mask)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
