import glob
import json
import re
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq

import time
class Timed:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        elapsed = end_time - self.start_time
        print(f"{self.name}: {elapsed:.4f} seconds")
        return False

def dump_parquet(texts: list[str], filename: str):
    arr = pa.array(texts)
    table = pa.Table.from_arrays([arr], names=['text'])
    pq.write_table(table, filename, compression='zstd', compression_level=3)
    # with Timed("snappy"):
    #     pq.write_table(table, f"ziptest-snappy-{filename}", compression='snappy')
    # with Timed("gzip"):
    #     pq.write_table(table, f"ziptest-gzip-{filename}", compression='gzip')
    # with Timed("zstd1"):
    #     pq.write_table(table, f"ziptest-zstd1-{filename}", compression='zstd', compression_level=1)
    # with Timed("zstd3"):
    #     pq.write_table(table, f"ziptest-zstd3-{filename}", compression='zstd', compression_level=3)
    # with Timed("zstd9"):
    #     pq.write_table(table, f"ziptest-zstd9-{filename}", compression='zstd', compression_level=9)
    #
    # with Timed("snappy"):
    #     _ = pq.read_table(f"ziptest-snappy-{filename}")
    # with Timed("gzip"):
    #     _ = pq.read_table(f"ziptest-gzip-{filename}")
    # with Timed("zstd1"):
    #     _ = pq.read_table(f"ziptest-zstd1-{filename}")
    # with Timed("zstd3"):
    #     _ = pq.read_table(f"ziptest-zstd3-{filename}")
    # with Timed("zstd9"):
    #     _ = pq.read_table(f"ziptest-zstd9-{filename}")

files = glob.glob("downloaded/20231101.en/*.parquet", recursive=True)
print(files)

chunk_idx = 0
dump = []
dump_size = 0
max_size = 268435456

patterns = [
    re.compile(r"^\s*(?:See also|References|External links|Further reading|Notes)\s*$", re.MULTILINE)
]

for file in files:
    print(file)
    r_buf = pq.read_table(file, columns=['text'])
    r_buf = [i.as_py().strip() for i in r_buf['text']]

    # for i in tqdm(r_buf):
    for i in r_buf:
        for pattern in patterns:
            m = pattern.search(i)
            if m:
                i = i[:m.start()]

        text_len = len(i.encode("utf-8"))
        if dump_size > 0 and text_len + dump_size > max_size:
            # with open(f"chunk-{chunk_idx:04}.json", "w", encoding="utf-8") as f:
            #     json.dump(dump, f, indent=0, ensure_ascii=False)
            dump_parquet(dump, f"chunk-{chunk_idx:04}.parquet")
            dump = []
            dump_size = 0
            chunk_idx += 1
            # exit()
        dump.append(i)
        dump_size += text_len

if len(dump) > 0:
    with open(f"chunk-{chunk_idx:04}.json", "w", encoding="utf-8") as f:
        json.dump(dump, f, indent=0, ensure_ascii=False)

