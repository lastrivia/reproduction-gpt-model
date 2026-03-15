import torch
from torch import nn
from torch.nn.functional import scaled_dot_product_attention


class Attention(nn.Module):
    def __init__(self, d_model, n_heads, rope_cache):
        super(Attention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.rope_sin, self.rope_cos = rope_cache
        # if d_model % (2 * n_heads) != 0:
        #     raise ValueError('d_model must be divisible by (2 * n_heads)')
        self.d_head = d_model // n_heads

        self.w_qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=True)

    def apply_rope(self, x, begin_pos=0):
        # input: [batch_size, self.n_heads, ctx_len, self.d_head]
        end_pos = begin_pos + x.shape[2]
        rope_sin = (self.rope_sin[begin_pos:end_pos, :])[None, None, :, :]
        rope_cos = (self.rope_cos[begin_pos:end_pos, :])[None, None, :, :]
        x0, x1 = x.chunk(2, dim=-1)
        x_rope = torch.cat([x0 * rope_cos - x1 * rope_sin,
                            x1 * rope_cos + x0 * rope_sin], dim=-1)
        return x_rope

    def forward(self, x):
        batch_size = x.shape[0]
        ctx_len = x.shape[1]

        qkv = self.w_qkv(x).view(batch_size, ctx_len, 3, self.d_model)

        q = qkv[:, :, 0, :].view(batch_size, ctx_len, self.n_heads, self.d_head).transpose(1, 2)
        k = qkv[:, :, 1, :].view(batch_size, ctx_len, self.n_heads, self.d_head).transpose(1, 2)
        v = qkv[:, :, 2, :].view(batch_size, ctx_len, self.n_heads, self.d_head).transpose(1, 2)
        # todo kv cache

        # apply RoPE
        q = self.apply_rope(q, begin_pos=0)
        k = self.apply_rope(k, begin_pos=0)

        out = scaled_dot_product_attention(
            q, k, v, is_causal=True
        ).transpose(1, 2).reshape(batch_size, ctx_len, self.d_model)

        return self.w_o(out)
