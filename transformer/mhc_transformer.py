import math
import torch
from torch import nn
from typing import Optional

from .kv_cache import KVCacheList
from .utils import get_module_class
from .hc_transformer import AttentionBlock, FFNBlock


def _logit(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x / (1.0 - x))


def sinkhorn(logits: torch.Tensor, iters: int) -> torch.Tensor:
    m = torch.exp(logits.float() - logits.float().amax(dim=(-2, -1), keepdim=True))
    for _ in range(iters):
        m = m / m.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        m = m / m.sum(dim=-2, keepdim=True).clamp_min(1e-12)
    return m.to(dtype=logits.dtype)


class MHCBlock(nn.Module):
    def __init__(
            self, *,
            block: nn.Module,
            norm: str,
            dim: int,
            expansion_rate: int,
            layer_id: int,
            sinkhorn_iters: int = 20,
            gate_init: float = 0.01,
            init_eps: float = 1e-4,
    ):
        super().__init__()

        self.block = block
        self.expansion_rate = expansion_rate
        self.dim = dim
        self.sinkhorn_iters = sinkhorn_iters

        pre = torch.full((expansion_rate,), init_eps)
        pre[layer_id % expansion_rate] = 1.0 - init_eps
        post = torch.zeros(expansion_rate)
        res = torch.eye(expansion_rate)
        if expansion_rate > 1:
            res = res * (1.0 - init_eps) + (1.0 - res) * (init_eps / (expansion_rate - 1))

        self.pre_bias = nn.Parameter(_logit(pre))
        self.post_bias = nn.Parameter(post)
        self.res_bias = nn.Parameter(res.log())

        out_dim = expansion_rate * expansion_rate + 2 * expansion_rate
        self.phi = nn.Parameter(torch.zeros(expansion_rate * dim, out_dim))
        self.alpha_pre = nn.Parameter(torch.ones(1) * gate_init)
        self.alpha_post = nn.Parameter(torch.ones(1) * gate_init)
        self.alpha_res = nn.Parameter(torch.ones(1) * gate_init)

        self.mhc_norm = get_module_class(norm)(expansion_rate * dim)

    def forward(self, h, **kwargs):
        # h: [B, T, expansion_rate, dim]

        h_flat = h.reshape(*h.shape[:-2], self.expansion_rate * self.dim)
        coeffs = self.mhc_norm(h_flat) @ self.phi
        raw_pre, raw_post, raw_res = torch.split(
            coeffs,
            [self.expansion_rate, self.expansion_rate, self.expansion_rate * self.expansion_rate],
            dim=-1,
        )

        raw_pre = self.alpha_pre * raw_pre + self.pre_bias
        raw_post = self.alpha_post * raw_post + self.post_bias
        raw_res = self.alpha_res * raw_res + self.res_bias.reshape(-1)

        h_pre = torch.sigmoid(raw_pre)
        h_post = 2.0 * torch.sigmoid(raw_post)
        h_res = sinkhorn(
            raw_res.reshape(*raw_res.shape[:-1], self.expansion_rate, self.expansion_rate),
            self.sinkhorn_iters,
        )

        h_layer_in = h_pre[..., None, :] @ h
        h_layer = self.block(h_layer_in.squeeze(-2), **kwargs)

        return h_res @ h + h_post[..., :, None] * h_layer[..., None, :]


class MHCTransformer(nn.Module):
    def __init__(
            self,
            *,
            n_layers: int,
            d_model: int,
            n_heads: int,
            vocab_size: int,
            norm: str,
            dropout: float,
            # Hyper-connection params
            expansion_rate: int,
            sinkhorn_iters: int = 10,
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
        self.sinkhorn_iters = sinkhorn_iters


        scale = 1.0 / expansion_rate ** 0.5

        # Layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=1.0 / math.sqrt(d_model))

        self.attn_blocks = nn.ModuleList([
            MHCBlock(
                block=AttentionBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    norm=norm,
                    dropout=dropout,
                    scale=scale
                ),
                norm=norm,
                dim=d_model,
                expansion_rate=expansion_rate,
                layer_id=i * 2,
                sinkhorn_iters=sinkhorn_iters,
            )
            for i in range(n_layers)
        ])
        self.ffn_blocks = nn.ModuleList([
            MHCBlock(
                block=FFNBlock(
                    d_model=d_model,
                    norm=norm,
                    dropout=dropout,
                    scale=scale
                ),
                norm=norm,
                dim=d_model,
                expansion_rate=expansion_rate,
                layer_id=i * 2 + 1,
                sinkhorn_iters=sinkhorn_iters,
            )
            for i in range(n_layers)
        ])

        self.final_norm = get_module_class(norm)(d_model)

    def forward(self, x: torch.Tensor, kv_cache: Optional[KVCacheList] = None) -> torch.Tensor:
        x = self.embedding(x)
        h = x.unsqueeze(-2).expand(-1, -1, self.expansion_rate, -1)

        for i in range(self.n_layers):
            h = self.attn_blocks[i](h, kv_cache=kv_cache[i] if kv_cache else None)
            h = self.ffn_blocks[i](h)

        x = self.final_norm(h.sum(dim=-2, keepdim=False))
        logits = x @ self.embedding.weight.T
        return logits


    def param_groups(
        self,
        weight_decay: float,
    ):
        decay = []
        no_decay = []

        hc_static_ids = set()

        for module in self.modules():
            if isinstance(module, MHCBlock):
                hc_static_ids.add(id(module.pre_bias))
                hc_static_ids.add(id(module.post_bias))
                hc_static_ids.add(id(module.res_bias))

        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue

            if id(p) in hc_static_ids:
                no_decay.append(p)
            elif p.ndim <= 1 or name.endswith(".bias"):
                # bias / norm / scalar gate
                no_decay.append(p)
            else:
                decay.append(p)

        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
