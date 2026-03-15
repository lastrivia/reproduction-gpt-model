import glob
import json
import re
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq

dataset_name = 'wikipedia'

def dump_parquet(texts: list[str], idx):
    arr = pa.array(texts)
    table = pa.Table.from_arrays([arr], names=['text'])
    pq.write_table(table, f"{dataset_name}/chunk-{idx:04}.parquet", compression='zstd', compression_level=3)

files = glob.glob(f"{dataset_name}/downloaded/20231101.en/*.parquet", recursive=True)
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

    for i in tqdm(r_buf):
        text_len = len(i.encode("utf-8"))
        if dump_size > 0 and text_len + dump_size > max_size:
            dump_parquet(dump, chunk_idx)
            dump = []
            dump_size = 0
            chunk_idx += 1
        dump.append(i)
        dump_size += text_len

if len(dump) > 0:
    dump_parquet(dump, chunk_idx)

