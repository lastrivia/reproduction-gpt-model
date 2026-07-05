## GPT Pretraining Reproduction

A compact PyTorch reproduction project for GPT-style pretraining. The repository implements a desktop-scale language model pipeline covering corpus processing, tokenizer-based binary compression, decoder-only Transformer pretraining, checkpointing, and training curve plotting.

### Overview

This project reproduces a desktop-scale GPT-style pretraining pipeline, with the model internals implemented using PyTorch tensors and modules.

Main features:

- Hand-written decoder-only Transformer components, including RoPE and KV cache support.
- Vanilla and Hyper-Connections Transformer variants.
- Hugging Face tokenizer-based tokenization.
- Dataset-local corpus indexing, downloading, chunking, shuffling, and tokenized binary compression scripts.
- YAML-based model/training presets with derived preset inheritance.
- Directory checkpoints with metadata, logs, curves, model state, and resumable optimizer/scheduler state.

### Model Presets

Presets live in `preset.yaml` and are loaded with `preset.py`. 

Common presets:

| Preset | Layers | Hidden | Heads | Norm | Seq len | Batch size | Tokens/step | VRAM (Vanilla) |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | 
| `smallest-classic` | 6 | 512 | 8 | LayerNorm | 512 | 16 | 8,192 | 9 GiB |
| `small-classic` | 12 | 768 | 12 | LayerNorm | 512 | 8 | 4,096 | 8 GiB |
| `medium-classic` | 24 | 1024 | 16 | LayerNorm | 512 | 4 | 2,048 | 11 GiB |
| `small` | 12 | 768 | 12 | RMSNorm | 2048 | 2 | 32,768 | 9 GiB |
| `small-server` | 12 | 768 | 12 | RMSNorm | 2048 | 16 | 32,768 | |
| `medium` | 24 | 1024 | 16 | RMSNorm | 2048 | 1 | 65,536 | 13 GiB |
| `medium-server` | 24 | 1024 | 16 | RMSNorm | 2048 | 8 | 65,536 | |

`residual_arch` selects the residual-stream architecture:

- `vanilla`: standard decoder-only Transformer residual stream, following Transformer-style residual blocks.
- `hc`: Hyper-Connections, which expand the residual stream into multiple streams and learn cross-depth feature routing ([Zhu et al., 2024](https://arxiv.org/abs/2409.19606)).
- `mhc`: *TODO.* Manifold-Constrained Hyper-Connections constrain the HC residual mixing space to improve identity mapping and efficiency ([Xie et al., 2025](https://arxiv.org/abs/2512.24880)).

Training uses AdamW with warmup, cosine decay, and a final constant learning-rate stage. Perplexity is logged every optimizer step, while console stats and checkpoints report average perplexity over the elapsed time interval since the previous stat/save event.

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

The current training entry point uses the prepared `fineweb-edu` dataset.

#### Pretraining

Start a new vanilla run by choosing a save directory and residual architecture:

```bash
python train.py -d results/vanilla -a vanilla
```

Start an HC run:

```bash
python train.py -d results/hc -p small -a hc --dynamic --tanh
```

Resume an existing checkpoint with the save directory:

```bash
python train.py -d results/hc
```

When `latest/` exists under the save directory, training arguments are automatically resumed from checkpoint metadata. CLI arguments are treated as optional consistency checks: explicitly provided arguments must match the checkpoint or training raises an error.

Supported training arguments:

```text
-d, --save-dir          Required checkpoint directory
-p, --model-preset      Preset name from preset.yaml
-a, --residual-arch     vanilla | hc; required only for a new run
--compatible            Use legacy sum-reduction loss scaling
--expansion-rate        HC-only expansion rate
--dynamic               HC-only dynamic routing flag
--tanh                  HC-only tanh flag
```

### Checkpoints

Each checkpoint is a directory under `save_dir`:

```text
latest/
finished/
bak_0/
bak_1/
...
```

`latest/` is the active resumable checkpoint. `finished/` marks a completed run and is treated as finished by the training entry point. Older checkpoints are rotated into `bak_*`.

### Source Structure

```text
.
|-- checkpoint.py          # Directory checkpoint save/load/init helpers
|-- dataset.py             # Single-dataset token batch IterableDataset
|-- dict_tools.py          # Recursive checked dict override/conflict helpers
|-- plot.py                # Training curve plotting utility
|-- preset.py              # Load and expand preset.yaml entries
|-- preset.yaml            # Model/training preset definitions
|-- scheduler.py           # Warmup/cosine/constant scheduler helper
|-- train.py               # Modern pretraining entry point
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
|   |-- attention.py       # Causal attention and KV cache attention
|   |-- rope.py            # RoPE positional encoding
|   |-- swiglu.py          # SwiGLU activation module
|   `-- kv_cache.py        # KV cache implementation for inference
|-- utils/                 # Utility scripts, including checkpoint/model conversion tools
`-- results/               # Checkpoints and training outputs
```

Evaluation and interactive inference are still TODO.
