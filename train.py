import argparse
import json
import math
import random
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
from transformer.transformer import Transformer


dataset_name = "fineweb-edu"
global_seed = 42
chinchilla_coeff = 20.0
weight_decay = 0.01
dropout = 0.1
stat_interval = 20
save_interval = 2000
save_ma_window = 100

device = torch.device("cuda:0")

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
}


def parse_args() -> tuple[str, dict]:
    parser = argparse.ArgumentParser(
        description="Train a GPT-style model.",
        argument_default=argparse.SUPPRESS,
    )

    parser.add_argument("-d", "--save-dir", type=Path, required=True)

    parser.add_argument("-p", "--model-preset")
    parser.add_argument("-a", "--residual-arch")

    parser.add_argument("--compatible", action="store_true")  # use sum-reduction for loss

    parser.add_argument("--expansion-rate", type=int)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--tanh", action="store_true")

    raw_args = parser.parse_args()
    save_dir = raw_args.save_dir
    raw_args = vars(raw_args)

    args = {
        name: raw_args[name]
        for name in raw_args
        if name in ("model_preset", "residual_arch", "compatible")
    }
    arch_params = {
        name: raw_args[name]
        for name in raw_args
        if name not in ("save_dir", "model_preset", "residual_arch", "compatible")
    }
    if arch_params:
        args["arch_params"] = arch_params

    return save_dir, args


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

    save_dir, cli_args = parse_args()

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

    tokenizer = AutoTokenizer.from_pretrained(
        "allenai/gpt-neox-olmo-dolma-v1_5",
        use_fast=True,
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

    warmup_steps = int(n_steps * 0.01)
    cosine_steps = int(n_steps * 0.95)

    save_size = total_params * 12
    save_times = n_steps // save_interval + 1
    print(f"Size of checkpoints: {save_size * save_times:,}")

    def build_optimizer(model) -> Optimizer:
        return AdamW(model.param_groups(weight_decay=weight_decay), lr=max_lr)

    def build_scheduler(optimizer):
        return build_warmup_cosine_scheduler(
            optimizer,
            warmup_steps=warmup_steps,
            cosine_steps=cosine_steps,
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

    def moving_average_loss_and_perplexity(rows: list[dict], window: int):
        recent_rows = rows[-window:]
        if not recent_rows:
            return None, None, 0
        ma_loss = sum(row["loss"] for row in recent_rows) / len(recent_rows)
        return ma_loss, math.exp(ma_loss), len(recent_rows)

    def save(finished: bool, step: int):
        ma_loss, ma_perplexity, ma_count = moving_average_loss_and_perplexity(log_rows, save_ma_window)
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
            "weight_decay": weight_decay,
            "dataset": dataset_name,
            "n_steps": n_steps,
            "training_tokens": actual_training_tokens,
            "ma_window": save_ma_window,
            "ma_count": ma_count,
            "ma_loss": ma_loss,
            "ma_perplexity": ma_perplexity,
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
        ma_label = f"{ma_perplexity:.2f}" if ma_perplexity is not None else "n/a"
        checkpoint_label = "Final checkpoint" if finished else "Checkpoint"
        print(f"{checkpoint_label} saved; MA{ma_count} perplexity: {ma_label}")

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

        if (step + 1) % stat_interval == 0:
            _, stat_ma_perplexity, stat_ma_count = moving_average_loss_and_perplexity(log_rows, stat_interval)
            ma_label = f"{stat_ma_perplexity:.2f}" if stat_ma_perplexity is not None else "n/a"
            print(f"Learning rate: {lr:.6f}, MA{stat_ma_count} perplexity: {ma_label}")

        if (step + 1) % save_interval == 0:
            save(finished=False, step=step)

        pbar.update(1)

    pbar.close()

    save(finished=True, step=n_steps - 1)


if __name__ == "__main__":
    main()
