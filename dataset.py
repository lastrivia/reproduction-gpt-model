import os
import glob
import random

import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset, get_worker_info, DataLoader
import zstandard as zstd
import numpy as np
from tqdm import tqdm


class ZstdTokenStreamDataset(IterableDataset):

    def __init__(self, files, seq_len, parallel_files=1, reader_streaming_size=(1<<20)):
        self.files = files
        self.seq_len = seq_len
        self.parallel = parallel_files
        self.reader_streaming_size = reader_streaming_size

    def _file_iter(self, file):
        # sliding window length: seq_len + 1
        #                step: seq_len

        seq_len = self.seq_len
        dctx = zstd.ZstdDecompressor()
        with open(file, "rb") as f:
            with dctx.stream_reader(f) as reader:

                bytes_buf = b''
                tokens_buf = np.empty(0, dtype=np.uint16)

                while True:
                    chunk = reader.read(self.reader_streaming_size)
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


    def __iter__(self):
        worker = get_worker_info()
        if worker is None:
            files = self.files
        else:
            files = self.files[worker.id:: worker.num_workers]

        file_idx = 0
        n_parallel = min(self.parallel, len(files))
        files_iter = [self._file_iter(file) for file in files[:n_parallel]]
        file_idx += n_parallel

        i = 0
        # round-robin
        while files_iter:
            if i >= len(files_iter):
                i = 0
            try:
                yield next(files_iter[i])
                i += 1
            except StopIteration:
                if file_idx < len(files):
                    files_iter[i] = self._file_iter(files[file_idx])
                    file_idx += 1
                else:
                    files_iter.pop(i)




if __name__ == "__main__":

    training_set = glob.glob("data/*/chunk-*-tokenized.bin.zst", recursive=True)
    print(len(training_set))
    random.seed(442)
    random.shuffle(training_set)
    print(len(training_set))

    sample = 100

    batch_size = 32
    max_len = 512

    print("steps (est): ", 60000000 * len(training_set) // batch_size // max_len)

    loader = DataLoader(
        dataset=ZstdTokenStreamDataset(files=training_set[:sample], seq_len=max_len, parallel_files=4),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        # prefetch_factor=2,
        # persistent_workers=True
    )

    steps = 0
    for batch in tqdm(loader):
        if steps == 0:
            print(batch.shape)
        steps += 1

    print("steps (est from sample): ", steps * len(training_set) // sample)
