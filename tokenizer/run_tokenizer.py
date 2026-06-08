import os
from pathlib import Path

import pyarrow.parquet as pq
import numpy as np
from transformers import AutoTokenizer
from multiprocessing import Pool
from tqdm import tqdm
from zstandard import ZstdCompressor

TOKENIZER_DIR = Path(__file__).resolve().parent
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

tokenizer = None
EOS = None
cctx: ZstdCompressor = None


def init_worker():
    global tokenizer
    global EOS
    global cctx
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_DIR / "tokenizer",
        use_fast=True,
        local_files_only=True,
    )
    tokenizer.model_max_length = 2 ** 30
    EOS = tokenizer.eos_token_id
    cctx = ZstdCompressor(level=3)


def tokenization_worker(file: str | Path):
    if os.name == "posix":
        os.nice(19)
    elif os.name == "nt":
        import psutil
        psutil.Process(os.getpid()).nice(psutil.IDLE_PRIORITY_CLASS)

    file = Path(file)
    save_file = file.parent.parent / "tokenized" / file.with_suffix(".bin.zst").name
    incomplete_file = save_file.with_name(save_file.name + ".INCOMPLETE")
    save_file.parent.mkdir(parents=True, exist_ok=True)
    if save_file.exists():
        return

    pf = pq.ParquetFile(file)
    n = pf.metadata.num_rows
    batch_size = max(n // 256, 16)

    if incomplete_file.exists():
        incomplete_file.unlink()

    with open(incomplete_file, "wb") as f:
        with cctx.stream_writer(f) as compressor:
            for batch in pf.iter_batches(columns=["text"], batch_size=batch_size):

                texts = batch["text"].to_pylist()
                encoded = tokenizer(
                    texts,
                    add_special_tokens=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )["input_ids"]

                lengths = [len(e) + 1 for e in encoded]
                tokens = np.empty(sum(lengths), dtype=np.uint32)
                pos = 0
                for e in encoded:
                    n = len(e)
                    tokens[pos:pos + n] = e
                    pos += n
                    tokens[pos] = EOS
                    pos += 1

                compressor.write(memoryview(tokens))

    os.replace(incomplete_file, save_file)


if __name__ == "__main__":

    test_tokenizer = AutoTokenizer.from_pretrained(
        "allenai/gpt-neox-olmo-dolma-v1_5",
        use_fast=True,
    )
    if test_tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer does not define eos_token_id")
    if len(test_tokenizer) > np.iinfo(np.uint32).max:
        raise ValueError(f"Tokenizer ids {len(test_tokenizer)} do not fit in uint32; update dataset.py before tokenizing")
    test_tokenizer.save_pretrained(TOKENIZER_DIR / "tokenizer")

    files = sorted(DATA_DIR.glob("*/shuffled/chunk-*.parquet"))

    with Pool(processes=4, initializer=init_worker) as pool:
        for _ in tqdm(
                pool.imap_unordered(tokenization_worker, files),
                total=len(files)
        ):
            pass

    # init_worker()
    # for file in tqdm(files):
    #     tokenization_worker(file)
