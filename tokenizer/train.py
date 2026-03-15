import pyarrow.parquet as pq
import glob
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from tqdm import tqdm

def iter_texts():
    files = sorted(glob.glob("../data/*/chunk-000?.parquet", recursive=True))
    print(files)

    for file in tqdm(files):
        table = pq.read_table(file, columns=["text"])
        col = table["text"]

        for x in col:
            yield x.as_py()

if __name__ == "__main__":

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=50000,
        min_frequency=2,
        special_tokens=[
            "<s>",
            "</s>",
            "<pad>",
            "<unk>"
        ]
    )

    tokenizer.train_from_iterator(iter_texts(), trainer)
    tokenizer.save("trained.json")