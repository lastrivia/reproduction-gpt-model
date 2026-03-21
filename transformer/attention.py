import torch
from torch import nn
from torch.nn.functional import scaled_dot_product_attention
from typing import Optional

from .rope import get_rope_cache, apply_rope
from .kv_cache import KVCacheLayer


class Attention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(Attention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        # if d_model % (2 * n_heads) != 0:
        #     raise ValueError('d_model must be divisible by (2 * n_heads)')
        self.d_head = d_model // n_heads

        self.w_qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=True)

    def forward(self, x: torch.Tensor, kv_cache: Optional[KVCacheLayer] = None) -> torch.Tensor:
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        qkv = self.w_qkv(x).view(batch_size, seq_len, 3, self.d_model)

        q = qkv[:, :, 0, :].view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = qkv[:, :, 1, :].view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = qkv[:, :, 2, :].view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        seq_offset = kv_cache.cache_len if kv_cache is not None else 0

        rope_cos, rope_sin = get_rope_cache(
            d_rope=self.d_head // 2,
            seq_len=seq_len,
            offset=seq_offset,
            device=x.device,
            dtype=x.dtype
        )
        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)

        if kv_cache is not None:
            kv_cache.append(k, v)
            k_cache, v_cache = kv_cache.get()

            if seq_len == 1:  # common inference
                out = scaled_dot_product_attention(
                    q, k_cache, v_cache, is_causal=False
                ).transpose(1, 2).reshape(batch_size, seq_len, self.d_model)

            else:
                attn_mask = torch.ones(seq_len, seq_offset + seq_len, device=x.device, dtype=torch.bool)
                attn_mask = torch.tril(attn_mask, diagonal=seq_offset)
                out = scaled_dot_product_attention(
                    q, k_cache, v_cache, attn_mask=attn_mask, is_causal=False
                ).transpose(1, 2).reshape(batch_size, seq_len, self.d_model)

        else:  # common training
            out = scaled_dot_product_attention(
                q, k, v, is_causal=True
            ).transpose(1, 2).reshape(batch_size, seq_len, self.d_model)

        return self.w_o(out)
