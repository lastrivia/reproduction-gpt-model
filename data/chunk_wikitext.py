import glob
import json
import re
from tqdm import tqdm

import pyarrow as pa
import pyarrow.parquet as pq

dataset_name = 'wikitext'

def dump_parquet(texts: list[str], idx):
    arr = pa.array(texts)
    table = pa.Table.from_arrays([arr], names=['text'])
    pq.write_table(table, f"{dataset_name}/chunk-{idx:04}.parquet", compression='zstd', compression_level=3)

files = glob.glob(f"{dataset_name}/downloaded/**/*.parquet", recursive=True)
print(files)

patterns = [
    re.compile(r"\s+([,.;:!?])"),
    re.compile(r"\s('[a-z]{1,2}\s)"),  # We ' re => We're
    re.compile(r'"\s([^\n]+?)\s"'),    # " Hello " => "Hello"
    re.compile(r"'\s([^\n]+?)\s'"),
    re.compile(r"\(\s([^\n]+?)\s\)"),
    re.compile(r"\[\s([^\n]+?)\s\]"),
    re.compile(r"s ' "),
    re.compile(r'\s@([^\n]{1,4}?)@\s'),
    re.compile(r'^\s*=\s([^=\n]+?)\s=*s*$'),  # = Caption =
    re.compile(r'(?:=\s)+([^\n]+?)\s*(?:\s=)+') # = = Section = =
]
repls = [
    r"\1",
    r"\1",
    r'"\1"',
    r"'\1'",
    r"(\1)",
    r"[\1]",
    r"s' ",
    r"\1",
    r"<SOT>\1",
    r"\1"
]

chunk_idx = 0
dump = []
dump_size = 0
max_size = 268435456

for file in files:
    print(file)
    r_buf = pq.read_table(file, columns=['text'])
    r_buf = [i.as_py().strip() for i in r_buf['text']]
    clean_text = []

    text_buf = ""
    for i in tqdm(r_buf):
        for pattern, repl in zip(patterns, repls):
            i = pattern.sub(repl, i)
        if i.startswith("<SOT>"):
            if len(text_buf) > 0:
                clean_text.append(text_buf)
                text_buf = ""
            i = i[len("<SOT>"):]
        if len(i) > 0:
            text_buf += i
            text_buf += "\n"
    if len(text_buf) > 0:
        clean_text.append(text_buf)

    for text in clean_text:
        text_len = len(text.encode("utf-8"))
        if text_len + dump_size > max_size:
            dump_parquet(dump, chunk_idx)
            dump = []
            dump_size = 0
            chunk_idx += 1
        dump.append(text)
        dump_size += text_len

if len(dump) > 0:
    dump_parquet(dump, chunk_idx)

