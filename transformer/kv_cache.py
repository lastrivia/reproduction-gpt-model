import torch
from typing import Tuple


class KVCacheLayer:
    def __init__(self, batch_size, n_heads, max_len, d_head, device, dtype):
        self.k = torch.empty(
            [batch_size, n_heads, max_len, d_head],
            device=device, dtype=dtype
        )
        self.v = torch.empty(
            [batch_size, n_heads, max_len, d_head],
            device=device, dtype=dtype
        )
        self.cache_len = 0
        self.max_len = max_len

    def append(self, k: torch.Tensor, v: torch.Tensor):
        if k.shape[2] != v.shape[2]:
            raise ValueError("lengths of k and v mismatch")
        if k.shape[2] + self.cache_len > self.max_len:
            raise OverflowError("KV cache overflow")
        self.k[:, :, self.cache_len:self.cache_len + k.shape[2], :] = k
        self.v[:, :, self.cache_len:self.cache_len + v.shape[2], :] = v
        self.cache_len += k.shape[2]

    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.k[:, :, :self.cache_len, :], self.v[:, :, :self.cache_len, :]

    def clear(self):
        self.cache_len = 0


class KVCache:
    def __init__(self, n_layers, batch_size, n_heads, max_len, d_head, device, dtype):
        self.layers = [
            KVCacheLayer(batch_size, n_heads, max_len, d_head, device, dtype)
            for _ in range(n_layers)
        ]

    def __getitem__(self, idx):
        return self.layers[idx]

    def clear(self):
        for layer in self.layers:
            layer.clear()
