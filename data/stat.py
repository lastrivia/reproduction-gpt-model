# import json
#
# file = "wikipedia/chunk-0000.json"
#
# with open(file, "r", encoding="utf-8") as f:
#     data = json.load(f)
#
# print(f"Items: {len(data)}")
#
# sum_len = sum([len(i.encode("utf-8")) for i in data])
#
# print(f"Average length: {sum_len / len(data)}")

import glob
import pyarrow as pa
import pyarrow.parquet as pq

files = glob.glob("*/chunk-0000.parquet", recursive=True)
for file in files:
    print(file)
    table = pq.read_table(file, columns=['text'])
    texts = [i.as_py() for i in table['text']]
    lengths = [len(i) for i in texts]

    print(f"Item count: {len(texts)}")
    print(f"Average length: {sum(lengths) / len(lengths)}")
    print()