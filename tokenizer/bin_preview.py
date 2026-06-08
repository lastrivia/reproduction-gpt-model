import zstandard as zstd
import numpy as np
from transformers import AutoTokenizer
# from tokenizers import Tokenizer
# from tokenizers.decoders import ByteLevel

EOS = 1

with open("../data/fineweb-edu/tokenized/chunk-0042.bin.zst", "rb") as f:
    dctx = zstd.ZstdDecompressor()
    with dctx.stream_reader(f) as reader:
        data = reader.read()

tokens_arr = np.frombuffer(data, dtype=np.uint32)
eos_pos = np.where(tokens_arr == EOS)[0]
print(eos_pos)

# tokenizer = Tokenizer.from_file("trained.json")
# tokenizer.decoder = ByteLevel()
tokenizer = AutoTokenizer.from_pretrained(
    "./tokenizer",
    use_fast=True,
    local_files_only=True,
)
start = 0
for pos in eos_pos:
    tokens = tokens_arr[start:pos].tolist()
    decoded = tokenizer.decode(tokens)
    print(decoded)
    input()
    start = pos + 1
