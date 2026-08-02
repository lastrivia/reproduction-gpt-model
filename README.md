## GPT Pretraining Reproduction

A compact PyTorch implementation of GPT-style pretraining, covering corpus preparation, training, and evaluation.

### Overview

Main features:

- Hand-written decoder-only Transformer components, including RoPE and KV cache support.
- Vanilla residual, Hyper-Connections (HC), and Manifold-Constrained Hyper-Connections (mHC) architectures.
- CUDA and Ascend NPU training through a shared accelerator interface.
- Distributed data parallel (DDP) training.
- Hugging Face tokenizer-based tokenization.
- Dataset downloading, chunking, shuffling, and tokenized binary compression scripts.
- YAML-based backbone model and training presets.
- Directory checkpoints with metadata, logs, curves, model state, and resumable training state.

### Model Presets

Backbone model and training presets live in `preset.yaml` and are loaded by `preset.py`.

| Preset | Layers | Hidden | Heads | Params | Seq length | Tokens/step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `small` | 12 | 768 | 12 | 152M | 2,048 | 32,768 |
| `medium` | 24 | 1,024 | 16 | 454M | 2,048 | 65,536 |

Supported residual-stream architectures include:

- `vanilla`: standard decoder-only Transformer residual stream.
- `hc`: Hyper-Connections, which expand the residual stream into multiple streams and learn cross-depth feature routing ([Zhu et al., 2024](https://arxiv.org/abs/2409.19606)).
- `mhc`: Manifold-Constrained Hyper-Connections, which constrain the HC residual mixing space to improve identity mapping and efficiency ([Xie et al., 2025](https://arxiv.org/abs/2512.24880)).

Training uses AdamW with warmup and cosine decay scheduling.

### Data Preparation

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

### Training

Start a new run:

```bash
python train.py <save_dir> -a <vanilla|hc|mhc> -p <small|medium> -b <micro_batch_size> [--cuda(default)|--cann]
```

For example:

```bash
python train.py results/vanilla -a vanilla -p small -b 1
```

When `latest/` exists under `save_dir`, model arguments can be restored from its checkpoint metadata. Resume a run by passing the same save directory:

```bash
python train.py results/vanilla -b 1
```

**Training arguments**

- `save_dir`: required on every run. Checkpoints are saved under this directory. Passing a directory containing `latest/` checkpoint resumes that run.

- **Model architecture arguments** must be provided when starting a new run. They are restored from checkpoint metadata when resuming.

  - `-a`, `--residual-arch <str>`: selects a residual architecture from `vanilla`, `hc`, or `mhc`.
  - `-p`, `--model-preset <str>`: selects a model preset in `preset.yaml`, such as `small` or `medium`.
  - HC-specific arguments:
    - `--expansion-rate <int>`: the residual-stream expansion rate.
    - `--dynamic`: enables dynamic HC.
    - `--tanh`: applies `tanh` in dynamic HC.
  - mHC-specific arguments:
    - `--expansion-rate <int>`: the residual-stream expansion rate.
    - `--sinkhorn-iters <int>`: the number of Sinkhorn-Knopp iterations.

- **Training device arguments** must be provided on every run according to the available accelerator type and VRAM capacity. They do not change the intended mathematical training behavior; actual results may still vary with accelerator implementations, numerical precision, and operation ordering.

  - `-b`, `--micro-batch-size <int>`: sequences per micro batch. Should be chosen to fit device VRAM.
  - `--cuda` (alias: `--nvidia`): use CUDA. Requires an NVIDIA GPU.
  - `--cann` (aliases: `--ascend`, `--huawei`): use CANN. Requires a Huawei Ascend NPU.

  If neither `--cuda` nor `--cann` is specified, `--cuda` is used by default. Specifying both results in an error.

  

**Distributed Training**

Start a 4-card CUDA run with `torchrun`:

```bash
torchrun --standalone --nproc-per-node=4 \
  train.py results/vanilla-ddp --cuda -a vanilla -p small -b 1
```

For Ascend NPU, select the CANN accelerator:

```bash
torchrun --standalone --nproc-per-node=4 \
  train.py results/vanilla-ddp --cann -a vanilla -p small -b 1
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

### Evaluation

`eval.py` evaluates a saved checkpoint and prints perplexity over a selected range of prepared batches:

```bash
python eval.py results/vanilla \
  --seq-len 2048 --batch-size 2 \
  --start-batch-idx 0 --n-batches 100
```

The checkpoint argument may point to a `model.pt` file, a checkpoint directory, or a training save directory.

Evaluation support for CANN is TODO.

### Source Structure

```text
.
|-- checkpoint.py                   # Checkpoint save/load helpers
|-- dataset.py                      # Dataset loader
|-- eval.py                         # Evaluation entry point
|-- preset.py                       # Preset loader
|-- preset.yaml                     # Model and training presets
|-- train.py                        # Training entry point
|-- accelerator/
|   |-- cuda.py                     # CUDA device and NCCL configuration
|   `-- cann.py                     # Ascend NPU device and HCCL configuration
|-- data/
|   |-- manifest.py                 # Chunk manifest generation
|   `-- */                          # Dataset preparation scripts and artifacts
|-- tokenizer/
|   |-- tokenizer/                  # Local saved Hugging Face tokenizer
|   `-- run_tokenizer.py            # Convert shuffled parquet chunks to token chunks
|-- transformer/
|   |-- transformer.py              # Vanilla Transformer
|   |-- hc_transformer.py           # Hyper-Connections Transformer
|   |-- mhc_transformer.py          # Manifold-Constrained HC Transformer
|   |-- attention.py                # Causal and KV-cache attention
|   |-- rope.py                     # Rotary position embeddings
|   |-- swiglu.py                   # SwiGLU activation
|   `-- kv_cache.py                 # Inference KV cache
`-- utils/
    |-- dict_tools.py               # Set operation helpers for Python dict
    |-- distributed_context.py      # Rank context for torchrun
    |-- fp_diagnosis.py             # Numerical diagnostics
    |-- plot.py                     # Training curve plotting
    `-- warmup_cosine_scheduler.py  # Scheduler
```
