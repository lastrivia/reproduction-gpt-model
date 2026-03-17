import math

import torch
from torch import nn
from typing import Optional, Tuple

from .attention import Attention
from .swiglu import SwiGLU


class DecoderBlock(nn.Module):
    def __init__(
            self,
            d_model: int, n_heads: int,
            dropout: float = 0.1
    ):
        super().__init__()

        self.norm_1 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, n_heads)
        self.dropout_1 = nn.Dropout(dropout)

        self.norm_2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 8),
            SwiGLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, rope_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        x = x + self.dropout_1(self.attn(self.norm_1(x), rope_cache=rope_cache))
        x = x + self.dropout_2(self.ffn(self.norm_2(x)))
        return x


class Transformer(nn.Module):
    def __init__(
            self,
            n_layers: int, d_model: int, n_heads: int,
            vocab_size: int, max_len: int
    ):
        super(Transformer, self).__init__()
        if d_model % (2 * n_heads) != 0:
            raise ValueError('d_model must be divisible by (2 * n_heads)')
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.vocab_size = vocab_size
        self.max_len = max_len

        # RoPE cache
        d_rope = self.d_head // 2
        angular_velocity = 1e-4 ** (torch.arange(d_rope) / d_rope)
        pos = torch.arange(max_len)
        angles = torch.outer(pos, angular_velocity)
        self.register_buffer("rope_sin", torch.sin(angles), persistent=False)
        self.register_buffer("rope_cos", torch.cos(angles), persistent=False)

        # Layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=1.0 / math.sqrt(d_model))
        self.decoders = nn.ModuleList([
            DecoderBlock(d_model=d_model, n_heads=n_heads)
            for _ in range(n_layers)
        ])
        self.final_ln = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.embedding(x)
        for decoder in self.decoders:
            x = decoder(x, rope_cache=(self.rope_sin, self.rope_cos))
        x = self.final_ln(x)
        logits = x @ self.embedding.weight.T
        return logits
