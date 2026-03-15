import glob
import json
import re
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq

dataset_name = "openwebtext2"


def dump_parquet(texts: list[str], idx):
    arr = pa.array(texts)
    table = pa.Table.from_arrays([arr], names=['text'])
    pq.write_table(table, f"{dataset_name}/chunk-{idx:04}.parquet", compression='zstd', compression_level=3)


if __name__ == "__main__":

    files = glob.glob(f"{dataset_name}/downloaded/raw/**/*.jsonl", recursive=True)
    print(files)

    chunk_idx = 0
    dump = []
    dump_size = 0
    max_size = 268435456

    # patterns = [
    #     re.compile(r"^\s*(?:See also|References|External links|Further reading|Notes)\s*$", re.MULTILINE)
    # ]
    from lang_filter import filter

    for i, file in zip(range(len(files)), files):
        print()
        print(f"({i + 1}/{len(files)})")
        print(file)
        r_buf = []
        with open(file, 'r', encoding='utf8') as f:
            for line in f:
                item = json.loads(line)
                if "content" in item:
                    r_buf.append(item["content"])

        filtered = filter(r_buf, n_proc=4)

        for i in filtered:
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
