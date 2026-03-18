import torch
from torch import nn
from torch.nn.functional import scaled_dot_product_attention

from .rope import get_rope_cache, apply_rope


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

    def forward(self, x):
        batch_size = x.shape[0]
        ctx_len = x.shape[1]

        qkv = self.w_qkv(x).view(batch_size, ctx_len, 3, self.d_model)

        q = qkv[:, :, 0, :].view(batch_size, ctx_len, self.n_heads, self.d_head).transpose(1, 2)
        k = qkv[:, :, 1, :].view(batch_size, ctx_len, self.n_heads, self.d_head).transpose(1, 2)
        v = qkv[:, :, 2, :].view(batch_size, ctx_len, self.n_heads, self.d_head).transpose(1, 2)
        # todo kv cache

        rope_cos, rope_sin = get_rope_cache(
            d_rope=self.d_head // 2,
            seq_len=ctx_len,
            offset=0,
            device=x.device,
            dtype=x.dtype
        )
        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)

        out = scaled_dot_product_attention(
            q, k, v, is_causal=True
        ).transpose(1, 2).reshape(batch_size, ctx_len, self.d_model)

        return self.w_o(out)
