import math
import torch
from torch import nn
from typing import Optional, Tuple

from .attention import Attention
from .swiglu import SwiGLU
from .kv_cache import KVCache, KVCacheList


class HyperConnectionBlock(nn.Module):
    def __init__(
            self,
            block: nn.Module,
            dim: int,
            expansion_rate: int,
            layer_id: int,
            dynamic: bool,
            tanh: bool
    ):
        super().__init__()
        
        self.block = block
        self.expansion_rate = expansion_rate
        self.dynamic = dynamic
        self.tanh = tanh

        # B [expansion_rate]
        self.b = nn.Parameter(torch.ones(expansion_rate))

        # A: Am & Ar [expansion_rate, expansion_rate + 1]
        am = torch.zeros(expansion_rate, 1)
        am[layer_id % expansion_rate, 0] = 1.0
        self.a = nn.Parameter(torch.cat([am, torch.eye(expansion_rate)], dim=1))

        if dynamic:
            self.w_a = nn.Parameter(torch.zeros(dim, expansion_rate + 1))
            self.s_a = nn.Parameter(torch.ones(1) * 0.01)

            self.w_b = nn.Parameter(torch.zeros(dim))
            self.s_b = nn.Parameter(torch.ones(1) * 0.01)

            self.layer_norm = nn.LayerNorm(dim)

    def forward(self, h, **kwargs):
        # h: [B, T, expansion_rate, dim]

        if self.dynamic:
            norm_h = self.layer_norm(h)

            if self.tanh:
                a = torch.tanh(norm_h @ self.w_a) * self.s_a
                b = torch.tanh(norm_h @ self.w_b) * self.s_b
            else:
                a = norm_h @ self.w_a * self.s_a
                b = norm_h @ self.w_b * self.s_b

            a = a + self.a[None, None, :, :]
            b = b + self.b[None, None, :]
        else:
            a = self.a[None, None, :, :]
            b = self.b[None, None, :]

        h_mix = a.transpose(-1, -2) @ h
        h_layer = self.block(h_mix[..., 0, :], **kwargs)

        return h_mix[..., 1:, :] + b[..., :, None] * h_layer[..., None, :]


class AttentionBlock(nn.Module):
    def __init__(
            self,
            d_model: int, n_heads: int,
            dropout: float = 0.1,
            scale: float = 1.0,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, n_heads)
        self.dropout = nn.Dropout(dropout)

        if scale != 1.0:
            with torch.no_grad():
                self.attn.w_o.weight.mul_(scale)

    def forward(self, x: torch.Tensor, kv_cache: Optional[KVCache] = None) -> torch.Tensor:
        return self.dropout(self.attn(self.norm(x), kv_cache=kv_cache))


class FFNBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            dropout: float = 0.1,
            scale: float = 1.0,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 8),
            SwiGLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.dropout = nn.Dropout(dropout)

        if scale != 1.0:
            with torch.no_grad():
                self.ffn[2].weight.mul_(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.ffn(self.norm(x)))


class HCTransformer(nn.Module):
    def __init__(
            self,
            n_layers: int, 
            d_model: int, 
            n_heads: int, 
            vocab_size: int,
            dropout: float = 0.1,
            # Hyper-connection params
            expansion_rate: int=2,
            dynamic: bool=False,
            tanh: bool=False
    ):
        super().__init__()

        if d_model % (2 * n_heads) != 0:
            raise ValueError('d_model must be divisible by (2 * n_heads)')  # d_rope == d_head // 2
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.vocab_size = vocab_size

        self.expansion_rate = expansion_rate
        self.dynamic = dynamic
        self.tanh = tanh


        scale = 1.0 / expansion_rate ** 0.5

        # Layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=1.0 / math.sqrt(d_model))
        
        self.attn_blocks = nn.ModuleList([
            HyperConnectionBlock(
                block=AttentionBlock(d_model, n_heads, dropout, scale),
                dim=d_model,
                expansion_rate=expansion_rate,
                layer_id=i * 2,
                dynamic=dynamic,
                tanh=tanh,
            )
            for i in range(n_layers)
        ])
        self.ffn_blocks = nn.ModuleList([
            HyperConnectionBlock(
                block=FFNBlock(d_model, dropout, scale),
                dim=d_model,
                expansion_rate=expansion_rate,
                layer_id=i * 2 + 1,
                dynamic=dynamic,
                tanh=tanh,
            )
            for i in range(n_layers)
        ])

        self.final_ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, kv_cache: Optional[KVCacheList] = None) -> torch.Tensor:
        x = self.embedding(x)
        h = x.unsqueeze(-2).expand(-1, -1, self.expansion_rate, -1)

        for i in range(self.n_layers):
            h = self.attn_blocks[i](h, kv_cache=kv_cache[i] if kv_cache else None)
            h = self.ffn_blocks[i](h)

        x = self.final_ln(h.sum(dim=-2, keepdim=False))
        logits = x @ self.embedding.weight.T
        return logits

def build_param_groups(model, weight_decay):
    decay = []
    no_decay = []

    hc_static_ids = set()
    for module in model.modules():
        if isinstance(module, HyperConnectionBlock):
            hc_static_ids.add(id(module.a))
            hc_static_ids.add(id(module.b))

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if id(p) in hc_static_ids:
            no_decay.append(p)
        elif p.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)

    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
