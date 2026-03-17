import zstandard as zstd
import numpy as np
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel

EOS = 1

with open("../data/wikipedia/chunk-0000-tokenized.bin.zst", "rb") as f:
    dctx = zstd.ZstdDecompressor()
    with dctx.stream_reader(f) as reader:
        data = reader.read()

tokens_arr = np.frombuffer(data, dtype=np.uint16)
eos_pos = np.where(tokens_arr == EOS)[0]
print(eos_pos)

tokenizer = Tokenizer.from_file("trained.json")
tokenizer.decoder = ByteLevel()
start = 0
for pos in eos_pos:
    tokens = tokens_arr[start:pos].tolist()
    decoded = tokenizer.decode(tokens)
    print(decoded)
    input()
    start = pos + 1
