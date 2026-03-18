from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class RopeCacheKey:
    d_rope: int
    device: torch.device
    dtype: torch.dtype


def next_pow2(x: int) -> int:
    return 1 << (x - 1).bit_length()


class RopeCache:
    def __init__(self):
        self.cache: dict[RopeCacheKey, tuple[torch.Tensor, torch.Tensor, int]] = {}

    def get(self, d_rope, seq_len, offset, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        # d_rope == d_head / 2
        # return shape (seq_len, d_rope)
        req_len = offset + seq_len
        key = RopeCacheKey(d_rope, device, dtype)

        do_rebuild = False
        if key not in self.cache:
            do_rebuild = True
        elif req_len > self.cache[key][2]:
            do_rebuild = True

        if do_rebuild:
            build_len = max(64, next_pow2(req_len))
            angular_v = 1e-4 ** (torch.arange(d_rope, device=device) / d_rope)
            idx = torch.arange(build_len, device=device)
            angles = torch.outer(idx, angular_v)
            self.cache[key] = (
                torch.cos(angles).to(dtype),
                torch.sin(angles).to(dtype),
                build_len
            )

        rope_cos, rope_sin, _ = self.cache[key]
        return rope_cos[offset:offset + seq_len, :], rope_sin[offset:offset + seq_len, :]

    def clear(self):
        self.cache.clear()


_GLOBAL_ROPE = RopeCache()


def get_rope_cache(d_rope, seq_len, offset, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    return _GLOBAL_ROPE.get(d_rope, seq_len, offset, device, dtype)


def clear_rope_cache():
    _GLOBAL_ROPE.clear()


def apply_rope(x, rope_cos, rope_sin):
    rope_cos = rope_cos.view((1,) * (x.dim() - 2) + rope_cos.shape)
    rope_sin = rope_sin.view((1,) * (x.dim() - 2) + rope_sin.shape)
    assert x.size(-1) & 1 == 0
    x_0, x_1 = x.chunk(2, dim=-1)
    return torch.cat([
        x_0 * rope_cos - x_1 * rope_sin,
        x_1 * rope_cos + x_0 * rope_sin
    ], dim=-1)
