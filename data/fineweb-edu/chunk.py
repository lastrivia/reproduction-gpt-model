from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent
input_dir = BASE_DIR / "downloaded"
output_dir = BASE_DIR / "chunk"
max_size = 1073741824
language = "en"


def dump_parquet(texts: list[pa.StringScalar], idx: int) -> None:
    table = pa.Table.from_pydict({"text": texts})
    output_file = output_dir / f"chunk-{idx:04}.parquet"
    pq.write_table(table, output_file, compression="zstd", compression_level=3)


files = sorted(input_dir.rglob("*.parquet"))
print(files)
output_dir.mkdir(parents=True, exist_ok=True)

chunk_idx = 0
dump = []
dump_size = 0

for i, file in enumerate(files):
    print(f"({i + 1}/{len(files)}) {file}")

    table = pq.read_table(file, columns=["text", "language"])
    table = table.filter(pc.equal(table["language"], language))
    text_arr = pc.utf8_trim_whitespace(table["text"])
    len_arr = pc.binary_length(text_arr)

    for text, text_len in tqdm(zip(text_arr, len_arr), total=len(text_arr)):
        text_len = text_len.as_py()

        if dump_size > 0 and text_len + dump_size > max_size:
            dump_parquet(dump, chunk_idx)
            dump = []
            dump_size = 0
            chunk_idx += 1

        dump.append(text)
        dump_size += text_len

if len(dump) > 0:
    dump_parquet(dump, chunk_idx)
    chunk_idx += 1

print(f"Wrote {chunk_idx} chunks")
