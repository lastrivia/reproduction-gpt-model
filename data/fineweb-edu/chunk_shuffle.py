from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent
input_dir = BASE_DIR / "chunk"
output_dir = BASE_DIR / "shuffled"
max_size = 1073741824
batch_size = 256


def dump_parquet(texts: list[pa.StringScalar], idx: int) -> None:
    table = pa.Table.from_pydict({"text": texts})
    output_file = output_dir / f"chunk-{idx:04}.parquet"
    pq.write_table(table, output_file, compression="zstd", compression_level=3)


files = sorted(input_dir.glob("chunk-*.parquet"))
print(files)
output_dir.mkdir(parents=True, exist_ok=True)

chunk_idx = 0
dump = []
dump_size = 0
readers = [
    pq.ParquetFile(file).iter_batches(
        batch_size=batch_size,
        columns=["text"],
    )
    for file in files
]

round_idx = 0
while len(readers) > 0:
    round_idx += 1
    print(f"Round {round_idx}: {len(readers)} active chunks")
    next_readers = []

    for reader in tqdm(readers):
        try:
            batch = next(reader)
        except StopIteration:
            continue

        table = pa.Table.from_batches([batch])
        text_arr = pc.utf8_trim_whitespace(table["text"])
        len_arr = pc.binary_length(text_arr)

        for text, text_len in zip(text_arr, len_arr):
            text_len = text_len.as_py()

            if dump_size > 0 and text_len + dump_size > max_size:
                dump_parquet(dump, chunk_idx)
                dump = []
                dump_size = 0
                chunk_idx += 1

            dump.append(text)
            dump_size += text_len

        next_readers.append(reader)

    readers = next_readers

if len(dump) > 0:
    dump_parquet(dump, chunk_idx)
    chunk_idx += 1

print(f"Wrote {chunk_idx} chunks")
