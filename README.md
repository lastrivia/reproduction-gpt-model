## GPT Pretraining Reproduction

A compact PyTorch reproduction project for GPT-style pretraining. The repository implements a desktop-scale language model pipeline covering corpus processing, Byte-Level BPE tokenizer training, decoder-only Transformer pretraining, checkpointing, and interactive text generation.

### Overview

This project reproduces a desktop-scale GPT-style pretraining pipeline, with the model internals implemented based on PyTorch Tensors and Modules.

Main features:

- Hand-written decoder-only Transformer components, including RoPE and KV Cache.
- Byte-Level BPE tokenizer training and tokenization.
- Corpus download, chunking, language filtering, and tokenized binary compression scripts.
- Weighted mixed-corpus streaming dataset for multiple English corpora.
- Preset model configurations from about 50M to 350M parameters.
- Checkpointing, training curve plotting, and interactive text generation.

### Usage

#### Data Preparation

1. Download corpora with `data/*/download.py`.
2. Convert raw corpora into parquet chunks with `data/chunk_*.py`.
3. Train a Byte-Level BPE tokenizer with `tokenizer/train.py`.
4. Convert parquet chunks into compressed token chunks with `tokenizer/tokenize.py`.
5. Use `dataset.py` to stream fixed-length token sequences from multiple datasets according to configured sampling weights.

The current training configuration mixes web, academic, literature, and wiki-style corpora, including C4, FineWeb, OpenWebText2, arXiv, PubMed, BookCorpus2, Wikipedia, and WikiText.

Train the tokenizer:

```bash
cd tokenizer
python train.py
```

#### Pre-training & Inference

Run pretraining or evaluation:

```bash
python train.py
```

Before running, check the local settings in `train.py`, especially `preset`, `do_train`, `global_no_save`, and `load_timestamp`.

Run interactive text generation:

```bash
python infer.py
```

### Model Presets

`train.py` provides three model presets:

| Preset | Layers | Hidden size | Heads | Approx. parameters |
| --- | ---: | ---: | ---: | ---: |
| `smallest` | 6 | 512 | 8 | 50M |
| `small` | 12 | 768 | 12 | 151M |
| `medium` | 18 | 1024 | 16 | 353M |

Training uses AdamW with warmup, cosine decay, and a final constant learning-rate stage. Perplexity is used as the main monitoring metric.

### Source Structure

```text
.
|-- data/                  # Data download, cleaning, chunking, and statistics scripts
|   |-- */download.py      # Dataset-specific download scripts
|   `-- chunk_*.py         # Convert raw corpora into parquet chunks
|-- tokenizer/             # Tokenizer training and tokenization tools
|   |-- train.py           # Train the Byte-Level BPE tokenizer
|   |-- tokenize.py        # Convert parquet text chunks into token chunks
|   `-- trained.json       # Trained tokenizer file
|-- transformer/           # Transformer model implementation
|   |-- transformer.py     # DecoderBlock and Transformer modules
|   |-- attention.py       # Causal attention and KV Cache attention
|   |-- rope.py            # RoPE positional encoding
|   |-- swiglu.py          # SwiGLU activation module
|   `-- kv_cache.py        # KV Cache implementation for inference
|-- weight/                # Checkpoints and training curve outputs
|-- dataset.py             # Mixed-corpus streaming Dataset
|-- train.py               # Pretraining and evaluation entry point
|-- infer.py               # Interactive text generation entry point
`-- plot.py                # Training curve plotting utility
```
