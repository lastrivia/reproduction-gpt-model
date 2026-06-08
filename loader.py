import bisect
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info
import zstandard as zstd

from data.manifest import load_manifest


DATA_DIR = Path(__file__).resolve().parent / "data"


class TokenizedBatchDataset(IterableDataset):
    def __init__(
            self,
            dataset: str,
            seq_len: int,
            batch_size: int,
            start_batch_idx: int = 0,
            max_batches: Optional[int] = None,
            rbuf_size: int = 1 << 20,
            print_file_ops: bool = False
    ):
        self.dataset = dataset
        self.dataset_dir = DATA_DIR / dataset
        self.tokenized_dir = self.dataset_dir / "tokenized"
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.start_batch_idx = start_batch_idx
        self.max_batches = max_batches
        self.rbuf_size = rbuf_size
        self.print_file_ops = print_file_ops

        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if start_batch_idx < 0:
            raise ValueError("start_batch_idx must be non-negative")
        if max_batches is not None and max_batches < 0:
            raise ValueError("max_batches must be non-negative")

        self.manifest = load_manifest(
            dataset,
            rbuf_size=rbuf_size,
            print_file_ops=print_file_ops,
            data_dir=DATA_DIR,
        )
        self.files = [
            self.tokenized_dir / item["path"]
            for item in self.manifest
        ]
        self.seq_counts = [
            max((item["tokens"] - 1) // self.seq_len, 0)
            for item in self.manifest
        ]
        self.seq_prefix = [0]
        for n_seq in self.seq_counts:
            self.seq_prefix.append(self.seq_prefix[-1] + n_seq)

        self.total_seq = self.seq_prefix[-1]
        self.total_batches = self.total_seq // self.batch_size
        self._cache_file = None
        self._cache_buf = None
        self._cache_tokens = None

    def _load_chunk(self, chunk_idx: int) -> np.ndarray:
        file = self.files[chunk_idx]
        if self._cache_file == file:
            return self._cache_tokens

        if self.print_file_ops:
            print(f"[Loader] Loading {file}")

        buf = bytearray()
        dctx = zstd.ZstdDecompressor()
        with open(file, "rb") as f:
            with dctx.stream_reader(f) as reader:
                while True:
                    chunk = reader.read(self.rbuf_size)
                    if not chunk:
                        break
                    buf.extend(chunk)

        if len(buf) % np.dtype(np.uint32).itemsize != 0:
            raise ValueError(f"tokenized chunk is not uint32-aligned: {file}")

        tokens = np.frombuffer(buf, dtype=np.uint32)
        self._cache_file = file
        self._cache_buf = buf
        self._cache_tokens = tokens
        return tokens

    def _read_batch(self, batch_idx: int) -> torch.Tensor:
        seq_idx = batch_idx * self.batch_size
        row = 0
        batch = np.empty((self.batch_size, self.seq_len + 1), dtype=np.int64)

        while row < self.batch_size:
            chunk_idx = bisect.bisect_right(self.seq_prefix, seq_idx) - 1
            local_seq_idx = seq_idx - self.seq_prefix[chunk_idx]
            n_seq = min(
                self.batch_size - row,
                self.seq_prefix[chunk_idx + 1] - seq_idx
            )

            tokens = self._load_chunk(chunk_idx)
            start = local_seq_idx * self.seq_len
            view = np.lib.stride_tricks.as_strided(
                tokens[start:start + (n_seq - 1) * self.seq_len + self.seq_len + 1],
                shape=(n_seq, self.seq_len + 1),
                strides=(self.seq_len * tokens.itemsize, tokens.itemsize),
            )
            batch[row:row + n_seq] = view

            row += n_seq
            seq_idx += n_seq

        return torch.from_numpy(batch)

    def __len__(self) -> int:
        end_batch_idx = self.total_batches
        if self.max_batches is not None:
            end_batch_idx = min(end_batch_idx, self.start_batch_idx + self.max_batches)
        return max(end_batch_idx - self.start_batch_idx, 0)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_cache_file"] = None
        state["_cache_buf"] = None
        state["_cache_tokens"] = None
        return state

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is not None and worker_info.num_workers > 1:
            raise ValueError("TokenizedBatchDataset supports num_workers=0 or 1 only")

        for batch_idx in range(self.start_batch_idx, self.start_batch_idx + len(self)):
            yield self._read_batch(batch_idx)
