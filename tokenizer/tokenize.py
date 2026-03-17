import os

import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import glob
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from zstandard import ZstdCompressor

EOS = 1

tokenizer: Tokenizer = None
cctx: ZstdCompressor = None


def init_worker():
    global tokenizer
    global cctx
    tokenizer = Tokenizer.from_file("trained.json")
    cctx = ZstdCompressor(level=3)


def tokenization_worker(file: str):
    save_file = file.replace(".parquet", "-tokenized.bin.zst")
    if os.path.exists(save_file):
        return

    pf = pq.ParquetFile(file)
    n = pf.metadata.num_rows
    batch_size = max(n // 256, 16)

    def token_iter(encoded):
        for e in encoded:
            for t in e.ids:
                yield t
            yield EOS

    if os.path.exists(save_file + ".INCOMPLETE"):
        os.remove(save_file + ".INCOMPLETE")
    with open(save_file + ".INCOMPLETE", "wb") as f:
        with cctx.stream_writer(f) as compressor:
            for batch in pf.iter_batches(columns=["text"], batch_size=batch_size):
                texts = batch["text"].to_pylist()
                encoded = tokenizer.encode_batch(texts, add_special_tokens=False)
                tokens = np.fromiter(token_iter(encoded), dtype=np.uint16)
                compressor.write(memoryview(tokens))
    os.rename(save_file + ".INCOMPLETE", save_file)


if __name__ == "__main__":
    files = glob.glob("../data/*/chunk-*.parquet", recursive=True)

    with Pool(processes=14, initializer=init_worker) as pool:
        for _ in tqdm(
                pool.imap_unordered(tokenization_worker, files),
                total=len(files)
        ):
            pass

    # init_worker()
    # for file in tqdm(files):
    #     tokenization_worker(file)
