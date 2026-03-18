import os
import glob
import random
import warnings
from typing import Optional, Generator
import torch
from torch.utils.data import IterableDataset, get_worker_info, DataLoader
import zstandard as zstd
import numpy as np
from torch.utils.data._utils.worker import WorkerInfo
from tqdm import tqdm


class MixedTokenStreamDataset(IterableDataset):

    def __init__(
            self,
            path: str,
            n_seq: int,
            seq_len: int,
            dataset_weights: dict[str, int],
            max_parallel: Optional[int] = None,
            rbuf_size: int = (1 << 20),
            print_file_ops: bool = False,
            seed: int = 42
    ):
        self.path = path
        self.n_seq = n_seq
        self.seq_len = seq_len
        self.dataset_weights = dataset_weights
        if max_parallel is None:
            self.max_parallel = len(dataset_weights)
        else:
            if max_parallel < len(dataset_weights):
                raise ValueError("max_parallel must be >= number of datasets")
            self.max_parallel = max_parallel
        self.rbuf_size = rbuf_size
        self.print_file_ops = print_file_ops
        self.seed = seed

    def _file_iter(self, file):
        # sliding window length: seq_len + 1
        #                step: seq_len

        seq_len = self.seq_len
        dctx = zstd.ZstdDecompressor()
        with open(file, "rb") as f:
            if self.print_file_ops:
                print(f"[Dataset] Reading {file}")
            with dctx.stream_reader(f) as reader:

                bytes_buf = b''
                tokens_buf = np.empty(0, dtype=np.uint16)

                while True:
                    chunk = reader.read(self.rbuf_size)
                    if not chunk:
                        break

                    # align with uint16
                    if bytes_buf:
                        chunk = bytes_buf + chunk
                    if len(chunk) & 1:
                        bytes_buf = chunk[-1:]
                        chunk = chunk[:-1]
                    else:
                        bytes_buf = b''

                    arr = np.frombuffer(chunk, dtype=np.uint16)

                    if tokens_buf.size:
                        concat_len = seq_len + 1 - tokens_buf.size
                        if arr.size < concat_len:
                            tokens_buf = np.concatenate((tokens_buf, arr))
                            continue
                        concat = np.concatenate((tokens_buf, arr[:concat_len]))
                        yield torch.tensor(concat, dtype=torch.long)
                        start_pos = concat_len - 1
                    else:
                        start_pos = 0

                    n = (arr.size - 1 - start_pos) // self.seq_len
                    for i in range(n):
                        yield torch.tensor(arr[
                                               start_pos + i * self.seq_len:
                                               start_pos + (i + 1) * self.seq_len + 1
                                           ], dtype=torch.long)

                    tokens_buf = arr[start_pos + n * self.seq_len:]
        if self.print_file_ops:
            print(f"[Dataset] Completed {file}")

    def _dataset_iter(self, dataset: str, n_parallel: int, worker_info=None):
        files = glob.glob(os.path.join(self.path, dataset, "chunk-*-tokenized.bin.zst"))
        if len(files) == 0:
            raise FileNotFoundError(f"dataset {dataset} not found")
        print(f"dataset: found {len(files)} chunks of dataset {dataset}")

        if worker_info is not None:
            worker_offset = worker_info.id
            if len(files) < worker_info.num_workers:
                warnings.warn(
                    f"{worker_info.num_workers} workers created on {len(files)} chunks of dataset {dataset}"
                )
                worker_offset %= worker_info.num_workers
            files = files[worker_offset::worker_info.num_workers]
        n_files = len(files)

        def shuffle_files(seed_offset):
            g = torch.Generator()
            g.manual_seed(self.seed ^ seed_offset)
            perm = torch.randperm(n_files, generator=g).tolist()
            return [files[i] for i in perm]

        shuffle_seed = 0
        shuffled_files = shuffle_files(shuffle_seed)

        load_idx = 0
        n_parallel = min(n_parallel, n_files)
        file_iters: list[Optional[Generator]] = [None for _ in range(n_parallel)]

        i = 0
        # round-robin, endless
        while True:
            if i >= n_parallel:
                i = 0
            if file_iters[i] is None:
                file_iters[i] = self._file_iter(shuffled_files[load_idx])
                try:
                    yield next(file_iters[i])
                    i += 1
                except StopIteration:
                    raise ValueError(f"invalid data chunk {shuffled_files[load_idx]}")
                load_idx = load_idx + 1
                if load_idx >= n_files:
                    load_idx = 0
                    shuffle_seed += 1
                    shuffled_files = shuffle_files(shuffle_seed)
            else:
                try:
                    yield next(file_iters[i])
                    i += 1
                except StopIteration:
                    file_iters[i] = None

    def __iter__(self):
        worker_info: Optional[WorkerInfo] = get_worker_info()

        dataset_weights = self.dataset_weights
        for _, value in dataset_weights.items():
            if value <= 0:
                raise ValueError(f"invalid dataset weight: {value}")

        # allocate n_parallel
        x = self.max_parallel - len(dataset_weights)
        total_weight = sum(dataset_weights.values())
        alloc_parallel = {
            key: 1 + x * weight // total_weight
            for key, weight in dataset_weights.items()
        }
        remain_n_parallel = self.max_parallel - sum(alloc_parallel.values())
        r = {
            key: x * weight % total_weight
            for key, weight in dataset_weights.items()
        }
        for key in sorted(dataset_weights.keys(), key=lambda k: r[k], reverse=True)[:remain_n_parallel]:
            alloc_parallel[key] += 1

        dataset_iters = {
            key : self._dataset_iter(key, alloc_parallel[key], worker_info)
            for key in dataset_weights.keys()
        }
        list_keys = list(dataset_weights.keys())
        list_weights = list(dataset_weights.values())
        for _ in range(self.n_seq):
            dataset = random.choices(list_keys, list_weights)[0]
            yield next(dataset_iters[dataset])


if __name__ == "__main__":

    training_set = glob.glob("data/*/chunk-*-tokenized.bin.zst", recursive=True)
    print(len(training_set))
    random.seed(442)
    random.shuffle(training_set)
    print(len(training_set))

    sample = 100

    batch_size = 32
    max_len = 512
    n_batches = 10000

    # print("steps (est): ", 60000000 * len(training_set) // batch_size // max_len)

    loader = DataLoader(
        dataset=MixedTokenStreamDataset(
            path="data",
            n_seq=n_batches * batch_size,
            seq_len=max_len,
            dataset_weights={
                "arxiv": 1
            },
            max_parallel=4
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        # prefetch_factor=2,
        # persistent_workers=True
    )

    steps = 0
    for batch in tqdm(loader, total=n_batches):
        if steps == 0:
            print(batch.shape)
        steps += 1

    # print("steps (est from sample): ", steps * len(training_set) // sample)
