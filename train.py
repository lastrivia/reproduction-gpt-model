import argparse
from contextlib import nullcontext
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as torch_dist
from torch.nn.functional import cross_entropy
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from checkpoint import CheckpointState, init_checkpoint, load_checkpoint_meta, save_checkpoint
from dataset import TokenizedBatchDataset
from utils.dict_tools import check_conflict, override_dict
from utils.warmup_cosine_scheduler import build_warmup_cosine_scheduler
from utils.distributed_context import DistributedContext
from preset import load_preset
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
save_interval_s = 1200.0
use_compile = True
check_fp_error = False

default_args = {
    "vanilla": {
        "model_preset": "small",
        "residual_arch": "vanilla",
    },
    "hc": {
        "model_preset": "small",
        "residual_arch": "hc",
        "arch_params": {
            "expansion_rate": 2,
            "dynamic": False,
            "tanh": False,
        },
    },
    "mhc": {
        "model_preset": "small",
        "residual_arch": "mhc",
        "arch_params": {
            "expansion_rate": 2,
            "sinkhorn_iters": 10,
        },
    },
}


def parse_args() -> tuple[Path, str, int | None, dict]:
    parser = argparse.ArgumentParser(
        description="Train a GPT-style model.",
        argument_default=argparse.SUPPRESS,
    )

    parser.add_argument("save_dir", type=Path)
    parser.add_argument("-b", "--micro-batch-size", type=int)

    accelerator_group = parser.add_mutually_exclusive_group()
    accelerator_group.add_argument(
        "--cuda",
        "--nvidia",
        dest="accelerator",
        action="store_const",
        const="cuda",
    )
    accelerator_group.add_argument(
        "--cann",
        "--ascend",
        "--huawei",
        dest="accelerator",
        action="store_const",
        const="cann",
    )
    parser.set_defaults(accelerator="cuda")

    parser.add_argument("-p", "--model-preset")
    parser.add_argument("-a", "--residual-arch")

    parser.add_argument("--expansion-rate", type=int)
    parser.add_argument("--sinkhorn-iters", type=int)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--tanh", action="store_true")

    raw_args = parser.parse_args()
    save_dir = raw_args.save_dir
    accelerator_name = raw_args.accelerator
    micro_batch_size = getattr(raw_args, "micro_batch_size", None)
    raw_args = vars(raw_args)

    args = {
        name: raw_args[name]
        for name in raw_args
        if name in ("model_preset", "residual_arch")
    }
    arch_params = {
        name: raw_args[name]
        for name in raw_args
        if name in ("expansion_rate", "sinkhorn_iters", "dynamic", "tanh")
    }
    if arch_params:
        args["arch_params"] = arch_params

    return save_dir, accelerator_name, micro_batch_size, args


def set_seed(seed: int, accelerator):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    accelerator.manual_seed(seed)


def main():
    # ================================
    # parse args
    # ================================

    save_dir, accelerator_name, cli_micro_batch_size, cli_args = parse_args()

    if accelerator_name == "cuda":
        from accelerator.cuda import Accelerator
    elif accelerator_name == "cann":
        from accelerator.cann import Accelerator
    else:
        raise NotImplementedError(f"unknown accelerator {accelerator_name}")

    dctx = DistributedContext.from_env()

    accelerator = Accelerator(dctx.local_rank)
    accelerator.set_device()

    set_seed(global_seed, accelerator)
    device = accelerator.device
    if dctx.is_main:
        print("Device:", accelerator.name)

    checkpoint_meta = load_checkpoint_meta(save_dir)
    meta = checkpoint_meta.meta

    if meta is None:  # Launch new training
        if "residual_arch" not in cli_args:
            raise ValueError("-a is required when starting without checkpoint")
        residual_arch = cli_args["residual_arch"]
        if residual_arch not in default_args:
            raise ValueError(f"unknown arch: {residual_arch}")
        run_args = override_dict(cli_args, default_args[residual_arch])

    else:  # Resume training
        checkpoint_args = {
            "model_preset": meta["model_preset"],
            "residual_arch": meta["residual_arch"],
        }
        if "arch_params" in default_args[checkpoint_args["residual_arch"]]:
            checkpoint_args["arch_params"] = meta["arch_params"]

        check_conflict(cli_args, checkpoint_args)
        run_args = checkpoint_args

    if dctx.is_main:
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
    seq_len = config["seq_len"]
    tokens_per_step = config["tokens_per_step"]
    max_lr = config["max_lr"]
    min_lr = config["min_lr"]
    warmup_ratio = config["warmup_ratio"]
    cosine_ratio = config["cosine_ratio"]

    micro_batch_size = (
        cli_micro_batch_size
        if cli_micro_batch_size is not None
        else config["default_micro_batch_size"]
    )
    if micro_batch_size <= 0:
        raise ValueError("micro_batch_size must be positive")
    if tokens_per_step % (micro_batch_size * seq_len) != 0:
        raise ValueError(
            "tokens_per_step must be divisible by micro_batch_size * seq_len: "
            f"{tokens_per_step} % ({micro_batch_size} * {seq_len}) != 0"
        )
    micro_batches_per_step = tokens_per_step // (micro_batch_size * seq_len)
    if micro_batches_per_step < dctx.world_size:
        raise ValueError(
            "micro batches per step must be at least world size: "
            f"{micro_batches_per_step} < {dctx.world_size}"
        )
    training_tokens = int(vanilla_params * chinchilla_coeff)
    total_steps = training_tokens // tokens_per_step
    actual_training_tokens = total_steps * tokens_per_step

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
    if dctx.is_main:
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
            scale_norm=True,
        )
    else:
        raise NotImplementedError(f"unknown residual_arch: {run_args['residual_arch']}")

    total_params = sum(p.numel() for p in model.parameters())
    if dctx.is_main:
        print(f"Params: {total_params:,}")
        print(f"Training steps: {total_steps:,}")
        print(f"Training tokens: {training_tokens:,} (target), {actual_training_tokens:,} (actual)")
        print(f"Micro batch size: {micro_batch_size}")
        print(f"Micro batches per step: {micro_batches_per_step}")

    # ================================
    # initialize training context
    # automatically resume from the latest checkpoint
    # ================================

    def build_optimizer(model) -> Optimizer:
        return AdamW(model.param_groups(weight_decay=weight_decay), lr=max_lr)

    def build_scheduler(optimizer):
        return build_warmup_cosine_scheduler(
            optimizer,
            warmup_steps=int(total_steps * warmup_ratio),
            cosine_steps=int(total_steps * cosine_ratio),
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
        if dctx.is_main:
            print(f"Checkpoint already finished: {state.checkpoint_dir}")
        return

    optimizer = state.optimizer
    scheduler = state.scheduler
    if optimizer is None or scheduler is None:
        raise RuntimeError("optimizer and scheduler must be initialized")

    start_step = state.meta["step"] + 1 if state.resumed else 0
    if start_step >= total_steps:
        if dctx.is_main:
            print("Training already reached target steps.")
        return

    # ================================
    # initialize data loader
    # ================================

    start_batch_idx = start_step * micro_batches_per_step
    max_batches = (total_steps - start_step) * micro_batches_per_step

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

    def save(
            finished: bool,
            step: int,
            avg_start_step: int,
            error: bool = False,
            diagnosis: dict | None = None,
    ):
        avg_loss, avg_perplexity, avg_count = average_loss_and_perplexity(avg_start_step, step)
        meta = run_args | {
            "n_layers": model.n_layers,
            "d_model": model.d_model,
            "n_heads": model.n_heads,
            "norm": norm,
            "seq_len": seq_len,
            "seed": global_seed,
            "tokens_per_step": tokens_per_step,
            "max_lr": max_lr,
            "min_lr": min_lr,
            "warmup_ratio": warmup_ratio,
            "cosine_ratio": cosine_ratio,
            "weight_decay": weight_decay,
            "grad_clip_norm": grad_clip_norm,
            "dataset": dataset_name,
            "n_steps": total_steps,
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
            error=error,
            diagnosis=diagnosis,
        )
        avg_label = f"{avg_perplexity:.2f}" if avg_perplexity is not None else "n/a"
        checkpoint_label = "Error checkpoint" if error else ("Final checkpoint" if finished else "Checkpoint")
        print(f"{checkpoint_label} saved; Avg perplexity: {avg_label} ({avg_count} steps)")

    model.train()
    train_model = model

    if use_compile:
        train_model = torch.compile(train_model)

    if dctx.enabled:
        torch_dist.init_process_group(backend=accelerator.distributed_backend)
        train_model = DistributedDataParallel(
            train_model,
            device_ids=[dctx.local_rank],
            output_device=dctx.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    optimizer.zero_grad()

    log_rows = state.log_rows

    data_iter = iter(loader)

    if dctx.is_main:
        pbar = tqdm(
            total=total_steps,
            initial=start_step,
            mininterval=0,
            ncols=80,
        )
        last_stat_time = time.monotonic()
        last_save_time = last_stat_time
        last_stat_step = start_step - 1
        last_save_step = start_step - 1

    try:
        for step in range(start_step, total_steps):

            local_loss_sum = torch.zeros((), device=device, dtype=torch.float32)

            for microbatch in range(micro_batches_per_step):
                batch = next(data_iter)
                if microbatch % dctx.world_size != dctx.rank:
                    continue
                batch = batch.to(device)
                inputs = batch[:, :-1]
                targets = batch[:, 1:]

                if dctx.enabled:
                    if microbatch + dctx.world_size < micro_batches_per_step:
                        sync_context = train_model.no_sync()
                    else:
                        sync_context = nullcontext()  # grad all-reduce
                else:
                    sync_context = nullcontext()

                with sync_context:
                    with accelerator.autocast(dtype=torch.bfloat16):
                        logits = train_model(inputs)
                        loss = cross_entropy(
                            logits.reshape(-1, logits.shape[-1]),
                            targets.reshape(-1),
                            reduction="mean",
                        )

                    loss_avg_scale = dctx.world_size / micro_batches_per_step
                    (loss * loss_avg_scale).backward()

                local_loss_sum += loss.detach().float()

            if dctx.enabled:
                torch_dist.reduce(local_loss_sum, dst=0, op=torch_dist.ReduceOp.SUM)
            if dctx.is_main:
                step_loss = (local_loss_sum / micro_batches_per_step).item()

            if check_fp_error:
                fp_error = not math.isfinite(step_loss) if dctx.is_main else False
                if dctx.enabled:  # broadcast error
                    fp_error_tensor = torch.tensor(
                        int(fp_error),
                        device=device,
                        dtype=torch.int32,
                    )
                    torch_dist.broadcast(fp_error_tensor, src=0)
                    fp_error = bool(fp_error_tensor.item())
                if fp_error:
                    optimizer.zero_grad()
                    if dctx.is_main:
                        raise FloatingPointError("bad step loss detected")
                    return

            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if dctx.is_main:
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
                    last_stat_time += math.floor((now - last_stat_time) / stat_interval_s + 0.5) * stat_interval_s
                    last_stat_step = step

                    if now - last_save_time >= save_interval_s:
                        save(finished=False, step=step, avg_start_step=last_save_step + 1)
                        last_save_time += math.floor((now - last_save_time) / save_interval_s + 0.5) * save_interval_s
                        last_save_step = step

                pbar.update(1)

        if dctx.is_main:
            pbar.close()
            save(finished=True, step=total_steps - 1, avg_start_step=last_save_step + 1)

    except FloatingPointError as exc:
        optimizer.zero_grad()
        if dctx.is_main:
            err_info = {
                "exception": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            }
            error_step = log_rows[-1]["step"] if log_rows else start_step - 1
            save(finished=False, step=error_step, avg_start_step=last_save_step + 1, error=True, diagnosis=err_info)
            pbar.close()
        raise

    finally:
        if dctx.enabled:
            torch_dist.destroy_process_group()


if __name__ == "__main__":
    main()
