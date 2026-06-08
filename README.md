## GPT Pretraining Reproduction

A compact PyTorch reproduction project for GPT-style pretraining. The repository implements a desktop-scale language model pipeline covering corpus processing, tokenizer-based binary compression, decoder-only Transformer pretraining, checkpointing, and training curve plotting.

### Overview

This project reproduces a desktop-scale GPT-style pretraining pipeline, with the model internals implemented based on PyTorch Tensors and Modules.

Main features:

- Hand-written decoder-only Transformer components, including RoPE and KV Cache.
- Hugging Face tokenizer-based tokenization.
- Dataset-local corpus indexing, downloading, chunking, shuffling, and tokenized binary compression scripts.
- Preset model configurations from about 50M to 350M parameters.
- Checkpointing, resumable single-dataset loading, and training curve plotting.

### Usage

#### Data Preparation

For each dataset under `data/*/`, run the dataset-local preparation scripts in order:

```bash
cd data/<dataset>
python index.py
python download.py
python chunk.py
python chunk_shuffle.py
```

Then tokenize the shuffled parquet chunks:

```bash
cd tokenizer
python run_tokenizer.py
```

The current training configuration uses a single prepared dataset, such as `fineweb-edu`.

#### Pre-training

Run pretraining:

```bash
python train.py
```

Before running, check the local settings in `train.py`, especially `model_preset`, `residual_arch`, `do_train`, `global_no_save`, and `load_timestamp`.

*Evaluation and the new interactive inference path are TODO.*

### Model Presets

`train.py` provides three model presets:

| Preset | Layers | Hidden size | Heads | Approx. parameters |
| --- | ---: | ---: | ---: | ---: |
| `smallest` | 6 | 512 | 8 | 50M |
| `small` | 12 | 768 | 12 | 151M |
| `medium` | 18 | 1024 | 16 | 353M |

`residual_arch` selects the residual-stream architecture:

- `vanilla`: standard decoder-only Transformer residual stream, following Transformer-style residual blocks.
- `hc`: Hyper-Connections, which expand the residual stream into multiple streams and learn cross-depth feature routing ([Zhu et al., 2024](https://arxiv.org/abs/2409.19606)).
- `mhc`: *TODO.* Manifold-Constrained Hyper-Connections constrain the HC residual mixing space to improve identity mapping and efficiency ([Xie et al., 2025](https://arxiv.org/abs/2512.24880)).

Training uses AdamW with warmup, cosine decay, and a final constant learning-rate stage. Perplexity is used as the main monitoring metric.

### Source Structure

```text
.
|-- data/                  # Data download, cleaning, chunking, and statistics scripts
|   |-- manifest.py        # Tokenized chunk manifest generation
|   `-- */                 # Dataset-specific index/download/chunk/shuffle scripts
|-- tokenizer/             # Tokenizer and tokenization tools
|   |-- tokenizer/         # Local saved Hugging Face tokenizer
|   |-- run_tokenizer.py   # Convert shuffled parquet chunks into token chunks
|   `-- bin_preview.py     # Inspect compressed token chunks
|-- transformer/           # Transformer model implementation
|   |-- transformer.py     # Vanilla DecoderBlock and Transformer modules
|   |-- hc_transformer.py  # HC-style Transformer variant
|   |-- attention.py       # Causal attention and KV Cache attention
|   |-- rope.py            # RoPE positional encoding
|   |-- swiglu.py          # SwiGLU activation module
|   `-- kv_cache.py        # KV Cache implementation for inference
|-- weight/                # Checkpoints and training curve outputs
|-- loader.py              # Single-dataset token batch loader
|-- train.py               # Pretraining entry point
`-- plot.py                # Training curve plotting utility
```
