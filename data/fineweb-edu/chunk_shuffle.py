from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import random
from tqdm import tqdm

seed = 42
random.seed(seed)

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
last_pbar_size = 0
dump_pbar = tqdm(
    total=max_size / 1e9,
    desc=f"chunk-{chunk_idx:04}.parquet",
    unit="GB",
    leave=True,
)
sources = []
for file in files:
    pf = pq.ParquetFile(file)
    rows = pf.metadata.num_rows
    sources.append({
        "file": file,
        "pf": pf,
        "batches": pf.iter_batches(
            batch_size=batch_size,
            columns=["text"],
        ),
        "remaining": rows,
    })

while len(sources) > 0:
    weights = [source["remaining"] for source in sources]
    source_idx = random.choices(range(len(sources)), weights=weights, k=1)[0]
    source = sources[source_idx]

    try:
        batch = next(source["batches"])
    except StopIteration:
        sources.pop(source_idx)
        continue

    source["remaining"] -= batch.num_rows

    table = pa.Table.from_batches([batch])
    text_arr = pc.utf8_trim_whitespace(table["text"])
    len_arr = pc.binary_length(text_arr)

    for text, text_len in zip(text_arr, len_arr):
        text_len = text_len.as_py()

        if dump_size > 0 and text_len + dump_size > max_size:
            dump_pbar.update((dump_size - last_pbar_size) / 1e9)
            dump_pbar.close()
            random.shuffle(dump)
            dump_parquet(dump, chunk_idx)

            dump = []
            dump_size = 0
            last_pbar_size = 0
            chunk_idx += 1
            dump_pbar = tqdm(
                total=max_size / 1e9,
                desc=f"chunk-{chunk_idx:04}.parquet",
                unit="GB",
                leave=True,
            )

        dump.append(text)
        dump_size += text_len

    dump_pbar.update((dump_size - last_pbar_size) / 1e9)
    last_pbar_size = dump_size

    if source["remaining"] <= 0:
        sources.pop(source_idx)

if len(dump) > 0:
    dump_pbar.update((dump_size - last_pbar_size) / 1e9)
    dump_pbar.close()
    random.shuffle(dump)
    dump_parquet(dump, chunk_idx)
    chunk_idx += 1
else:
    dump_pbar.close()

print(f"Wrote {chunk_idx} chunks")
