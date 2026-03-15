import glob
import json
import re
from tqdm import tqdm
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import gzip

dataset_name = "fineweb"


def dump_parquet(texts: list[str] | list[pa.StringScalar], idx):
    if isinstance(texts[0], str):
        arr = pa.array(texts)
        table = pa.Table.from_arrays([arr], names=['text'])
    elif isinstance(texts[0], pa.StringScalar):
        table = pa.Table.from_pydict({"text": texts})
    else:
        raise TypeError
    pq.write_table(table, f"{dataset_name}/chunk-{idx:04}.parquet", compression='zstd', compression_level=3)


if __name__ == "__main__":

    files = glob.glob(f"{dataset_name}/downloaded/**/*.parquet", recursive=True)
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
        print(f"({i + 1}/{len(files)}) {file}")
        r_buf = []

        table = pq.read_table(file, columns=['text', 'language'])
        table = table.filter(pc.equal(table['language'], 'en'))

        text_arr = pc.utf8_trim_whitespace(table['text'])
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
