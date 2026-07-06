import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import cross_entropy
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from checkpoint import CheckpointState, init_checkpoint, load_checkpoint_meta, save_checkpoint
from dataset import TokenizedBatchDataset
from dict_tools import check_conflict, override_dict
from preset import load_preset
from scheduler import build_warmup_cosine_scheduler
from transformer.hc_transformer import HCTransformer
from transformer.mhc_transformer import MHCTransformer
from transformer.transformer import Transformer


dataset_name = "fineweb-edu"
global_seed = 42
chinchilla_coeff = 20.0
weight_decay = 0.01
dropout = 0.1
grad_clip_norm = 1.0
stat_interval_s = 30.0
save_interval_s = 1800.0

default_args = {
    "vanilla": {
        "model_preset": "small",
        "residual_arch": "vanilla",
        "compatible": False,
    },
    "hc": {
        "model_preset": "small",
        "residual_arch": "hc",
        "compatible": False,
        "arch_params": {
            "expansion_rate": 2,
            "dynamic": False,
            "tanh": False,
        },
    },
    "mhc": {
        "model_preset": "small",
        "residual_arch": "mhc",
        "compatible": False,
        "arch_params": {
            "expansion_rate": 2,
            "sinkhorn_iters": 10,
        },
    },
}


def parse_args() -> tuple[Path, int, dict]:
    parser = argparse.ArgumentParser(
        description="Train a GPT-style model.",
        argument_default=argparse.SUPPRESS,
    )

    parser.add_argument("-d", "--save-dir", type=Path, required=True)
    parser.add_argument("-c", "--cuda", type=int, default=0)

    parser.add_argument("-p", "--model-preset")
    parser.add_argument("-a", "--residual-arch")

    parser.add_argument("--compatible", action="store_true")  # legacy training implementation

    parser.add_argument("--expansion-rate", type=int)
    parser.add_argument("--sinkhorn-iters", type=int)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--tanh", action="store_true")

    raw_args = parser.parse_args()
    save_dir = raw_args.save_dir
    cuda_id = raw_args.cuda
    raw_args = vars(raw_args)

    args = {
        name: raw_args[name]
        for name in raw_args
        if name in ("model_preset", "residual_arch", "compatible")
    }
    arch_params = {
        name: raw_args[name]
        for name in raw_args
        if name not in ("save_dir", "cuda", "model_preset", "residual_arch", "compatible")
    }
    if arch_params:
        args["arch_params"] = arch_params

    return save_dir, cuda_id, args


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    set_seed(global_seed)

    # ================================
    # parse args
    # ================================

    save_dir, cuda_id, cli_args = parse_args()
    device = torch.device(f"cuda:{cuda_id}")
    print("Device:", device)

    checkpoint_meta = load_checkpoint_meta(save_dir)
    meta = checkpoint_meta.meta

    if meta is None:
        if "residual_arch" not in cli_args:
            raise ValueError("-a is required when starting without checkpoint")
        residual_arch = cli_args["residual_arch"]
        if residual_arch not in default_args:
            raise ValueError(f"unknown arch: {residual_arch}")
        run_args = override_dict(cli_args, default_args[residual_arch])

    else:
        checkpoint_args = {
            "model_preset": meta["model_preset"],
            "residual_arch": meta["residual_arch"],
            "compatible": meta["compatible"],
        }
        if "arch_params" in default_args[checkpoint_args["residual_arch"]]:
            checkpoint_args["arch_params"] = meta["arch_params"]

        check_conflict(cli_args, checkpoint_args)
        run_args = checkpoint_args

    print("Run args:")
    print(json.dumps(run_args, indent=2))

    # ================================
    # load training preset
    # ================================

    config = load_preset(run_args["model_preset"])
    n_layers = config["n_layers"]
    d_model = config["d_model"]
    n_heads = config["n_heads"]
    norm = config["norm"]
    vanilla_params = config["vanilla_params"]
    micro_batch_size = config["micro_batch_size"]
    seq_len = config["seq_len"]
    tokens_per_step = config["tokens_per_step"]
    max_lr = config["max_lr"]
    min_lr = config["min_lr"]
    warmup_ratio = config["warmup_ratio"]
    cosine_ratio = config["cosine_ratio"]

    if tokens_per_step % (micro_batch_size * seq_len) != 0:
        raise ValueError(
            "invalid preset: tokens_per_step must be divisible by "
            "micro_batch_size * seq_len"
        )
    grad_accum_steps = tokens_per_step // (micro_batch_size * seq_len)
    training_tokens = int(vanilla_params * chinchilla_coeff)
    n_steps = training_tokens // tokens_per_step
    actual_training_tokens = n_steps * tokens_per_step

    # ================================
    # prepare tokenizer
    # ================================

    # tokenizer = AutoTokenizer.from_pretrained(
    #     "allenai/gpt-neox-olmo-dolma-v1_5",
    #     use_fast=True,
    # )
    tokenizer = AutoTokenizer.from_pretrained(
        "./tokenizer/tokenizer",
        use_fast=True,
        local_files_only=True
    )
    vocab_size = len(tokenizer)
    print("Vocab size:", vocab_size)

    # ================================
    # initialize model
    # ================================

    common_model_kwargs = {
        "n_layers": n_layers,
        "d_model": d_model,
        "n_heads": n_heads,
        "vocab_size": vocab_size,
        "norm": norm,
        "dropout": dropout,
    }
    if run_args["residual_arch"] == "vanilla":
        model = Transformer(**common_model_kwargs)
    elif run_args["residual_arch"] == "hc":
        model = HCTransformer(
            **common_model_kwargs,
            **run_args["arch_params"],
        )
    elif run_args["residual_arch"] == "mhc":
        model = MHCTransformer(
            **common_model_kwargs,
            **run_args["arch_params"],
        )
    else:
        raise NotImplementedError(f"unknown residual_arch: {run_args['residual_arch']}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {total_params:,}")
    print(f"Training steps: {n_steps:,}")
    print(f"Training tokens: {training_tokens:,} (target), {actual_training_tokens:,} (actual)")
    print(f"Gradient accumulation steps: {grad_accum_steps}")

    # ================================
    # initialize training context
    # automatically resume from the latest checkpoint
    # ================================

    def build_optimizer(model) -> Optimizer:
        return AdamW(model.param_groups(weight_decay=weight_decay), lr=max_lr)

    def build_scheduler(optimizer):
        return build_warmup_cosine_scheduler(
            optimizer,
            warmup_steps=int(n_steps * warmup_ratio),
            cosine_steps=int(n_steps * cosine_ratio),
            max_lr=max_lr,
            min_lr=min_lr,
        )

    state: CheckpointState = init_checkpoint(
        save_dir=save_dir,
        model=model,
        device=device,
        build_optimizer=build_optimizer,
        build_scheduler=build_scheduler,
    )
    if state.finished:
        print(f"Checkpoint already finished: {state.checkpoint_dir}")
        return

    optimizer = state.optimizer
    scheduler = state.scheduler
    if optimizer is None or scheduler is None:
        raise RuntimeError("optimizer and scheduler must be initialized")

    start_step = state.meta["step"] + 1 if state.resumed else 0
    if start_step >= n_steps:
        print("Training already reached target steps.")
        return

    # ================================
    # initialize data loader
    # ================================

    start_batch_idx = start_step * grad_accum_steps
    max_batches = (n_steps - start_step) * grad_accum_steps

    loader = DataLoader(
        TokenizedBatchDataset(
            dataset=dataset_name,
            seq_len=seq_len,
            batch_size=micro_batch_size,
            start_batch_idx=start_batch_idx,
            max_batches=max_batches,
        ),
        batch_size=None,
        num_workers=1,
        prefetch_factor=2,
        persistent_workers=True,
        pin_memory=True,
    )

    # ================================
    # run training
    # ================================

    def average_loss_and_perplexity(start_step: int, end_step: int):
        count = end_step - start_step + 1
        if count <= 0:
            return None, None, 0
        rows = log_rows[start_step:end_step + 1]
        avg_loss = sum(row["loss"] for row in rows) / count
        return avg_loss, math.exp(avg_loss), count

    def save(finished: bool, step: int, avg_start_step: int):
        avg_loss, avg_perplexity, avg_count = average_loss_and_perplexity(avg_start_step, step)
        meta = run_args | {
            "n_layers": model.n_layers,
            "d_model": model.d_model,
            "n_heads": model.n_heads,
            "norm": norm,
            "seq_len": seq_len,
            "seed": global_seed,
            "micro_batch_size": micro_batch_size,
            "tokens_per_step": tokens_per_step,
            "grad_accum_steps": grad_accum_steps,
            "max_lr": max_lr,
            "min_lr": min_lr,
            "warmup_ratio": warmup_ratio,
            "cosine_ratio": cosine_ratio,
            "weight_decay": weight_decay,
            "grad_clip_norm": None if run_args["compatible"] else grad_clip_norm,
            "dataset": dataset_name,
            "n_steps": n_steps,
            "training_tokens": actual_training_tokens,
            "stat_interval_s": stat_interval_s,
            "save_interval_s": save_interval_s,
            "avg_count": avg_count,
            "avg_loss": avg_loss,
            "avg_perplexity": avg_perplexity,
            "finished": finished,
            "step": step,
        }
        save_checkpoint(
            save_dir=save_dir,
            meta=meta,
            log_rows=log_rows,
            model=model,
            optimizer=None if finished else optimizer,
            scheduler=None if finished else scheduler,
            finished=finished,
        )
        avg_label = f"{avg_perplexity:.2f}" if avg_perplexity is not None else "n/a"
        checkpoint_label = "Final checkpoint" if finished else "Checkpoint"
        print(f"{checkpoint_label} saved; Avg perplexity: {avg_label} ({avg_count} steps)")

    model.train()
    optimizer.zero_grad()

    log_rows = state.log_rows

    pbar = tqdm(
        total=n_steps,
        initial=start_step,
        mininterval=0,
        ncols=80,
    )
    data_iter = iter(loader)

    last_stat_time = time.monotonic()
    last_save_time = last_stat_time
    last_stat_step = start_step - 1
    last_save_step = start_step - 1

    for step in range(start_step, n_steps):
        step_loss = 0.0

        for _ in range(grad_accum_steps):
            batch = next(data_iter).to(device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(inputs)
                loss = cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    targets.reshape(-1),
                    reduction="sum" if run_args["compatible"] else "mean",
                )
                (loss / grad_accum_steps).backward()

            micro_loss = loss.item()
            if run_args["compatible"]:
                micro_loss /= targets.numel()
            step_loss += micro_loss / grad_accum_steps

        if not run_args["compatible"]:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        ppl = math.exp(step_loss)
        lr = optimizer.param_groups[0]["lr"]

        log_rows.append({
            "step": step,
            "loss": step_loss,
            "ppl": ppl,
            "lr": lr,
        })

        now = time.monotonic()
        if now - last_stat_time >= stat_interval_s:
            _, stat_avg_perplexity, stat_avg_count = average_loss_and_perplexity(last_stat_step + 1, step)
            avg_label = f"{stat_avg_perplexity:.2f}" if stat_avg_perplexity is not None else "n/a"
            print(f"Learning rate: {lr:.6f}, Avg perplexity: {avg_label} ({stat_avg_count} steps)")
            last_stat_time = now
            last_stat_step = step

            if now - last_save_time >= save_interval_s:
                save(finished=False, step=step, avg_start_step=last_save_step + 1)
                last_save_time = time.monotonic()
                last_save_step = step

        pbar.update(1)

    pbar.close()

    save(finished=True, step=n_steps - 1, avg_start_step=last_save_step + 1)


if __name__ == "__main__":
    main()
