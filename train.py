import math
import time
import os
import numpy as np
import json
import random
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, ConstantLR
from torch.nn.functional import cross_entropy
from transformers import AutoTokenizer

from transformer.transformer import Transformer
from transformer.hc_transformer import HCTransformer
from loader import TokenizedBatchDataset
from plot import plot_training_curve


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


global_no_save = False
do_train = True
load_timestamp = None
# load_timestamp = "0319-224725"

model_preset = "small"  # smallest | small | medium
residual_arch = "vanilla"  # vanilla | hc | mhc
save_dir = "weight/small-vanilla-0608"

hc_expansion_rate = 2
hc_dynamic = False
hc_tanh = False


def save_checkpoint(
        meta, 
        model, 
        optimizer=None, 
        scheduler=None,
        ppl_plot_x=None, 
        ppl_plot_y=None, 
        show_plt=False,
):
    if global_no_save:
        if ppl_plot_x is not None and ppl_plot_y is not None:
            plot_training_curve(
                iteration=ppl_plot_x,
                ppl=ppl_plot_y,
                show=show_plt
            )
        return

    os.makedirs(save_dir, exist_ok=True)

    timestamp = time.strftime("%m%d-%H%M%S")
    with open(f"{save_dir}/gpt-{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)
    torch.save(model.state_dict(), f"{save_dir}/gpt-{timestamp}.pt")

    if optimizer is not None:
        torch.save(optimizer.state_dict(), f"{save_dir}/gpt-{timestamp}.opt.pt")
    if scheduler is not None:
        torch.save(scheduler.state_dict(), f"{save_dir}/gpt-{timestamp}.sch.pt")

    if ppl_plot_x is not None and ppl_plot_y is not None:
        plot_training_curve(
            iteration=ppl_plot_x,
            ppl=ppl_plot_y,
            show=show_plt,
            save=f"{save_dir}/gpt-{timestamp}-curve.png"
        )

    print(f"Checkpoint {timestamp} saved; Avg perplexity: {meta['avg_perplexity']}")


def preset(name):  # n_layers, d_model, n_heads, batch_size, max_lr, min_lr
    match name:
        case "smallest":  # 50M params,  9G VRAM
            return 6, 512, 8, 16, 5e-4, 5e-5
        case "small":  # 151M params, 8G VRAM
            return 12, 768, 12, 8, 2e-4, 2e-5
        case "medium":  # 353M params, 9G VRAM
            return 18, 1024, 16, 4, 1e-4, 1e-5
    raise NotImplementedError


if __name__ == "__main__":
    global_seed = 42
    set_seed(global_seed)

    tokenizer = AutoTokenizer.from_pretrained(
        "allenai/dolma2-tokenizer",
        use_fast=True,
    )
    vocab_size = len(tokenizer)
    print("Vocab size:", vocab_size)

    n_layers, d_model, n_heads, batch_size, max_lr, min_lr = preset(model_preset)
    weight_decay = 0.01
    seq_len = 512

    if residual_arch == "vanilla":
        model = Transformer(
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            vocab_size=vocab_size,
            dropout=0.1,
        )
    elif residual_arch == "hc":
        model = HCTransformer(
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            vocab_size=vocab_size,
            dropout=0.1,
            expansion_rate=hc_expansion_rate,
            dynamic=hc_dynamic,
            tanh=hc_tanh,
        )
    else:
        raise NotImplementedError
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {total_params:,}")

    device = torch.device("cuda:0")

    training_token_coeff = 20.0
    n_batches = int(total_params * training_token_coeff / batch_size / seq_len)
    n_tokens = n_batches * batch_size * seq_len
    print(f"Training tokens: {n_tokens:,}")

    warmup_steps = int(n_batches * 0.01)
    cosine_steps = int(n_batches * 0.95)

    stat_interval = 200
    save_interval = 20000

    save_size = total_params * 12
    save_times = n_batches // save_interval + 1 if not global_no_save else 0
    print(f"Size of checkpoints: {save_size * save_times:,}")


    if load_timestamp is None:
        # Train from scratch
        model.to(device)

        optimizer = AdamW(model.param_groups(weight_decay=weight_decay), lr=max_lr)

        scheduler = SequentialLR(
            optimizer=optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=1e-5, end_factor=1.0, total_iters=warmup_steps),  # 1 %
                CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=min_lr),  # 95 %
                ConstantLR(optimizer, factor=min_lr / max_lr, total_iters=n_batches * 999)  # rest
            ],
            milestones=[warmup_steps, warmup_steps + cosine_steps]
        )

        start_batch_idx = 0

    else:
        # Load checkpoint
        model.load_state_dict(torch.load(f"{save_dir}/gpt-{load_timestamp}.pt"))
        model.to(device)

        optimizer = AdamW(model.param_groups(weight_decay=weight_decay), lr=max_lr)
        optimizer.load_state_dict(torch.load(f"{save_dir}/gpt-{load_timestamp}.opt.pt"))

        scheduler = SequentialLR(
            optimizer=optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=1e-5, end_factor=1.0, total_iters=warmup_steps),  # 1 %
                CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=min_lr),  # 95 %
                ConstantLR(optimizer, factor=min_lr / max_lr, total_iters=n_batches * 999)  # rest
            ],
            milestones=[warmup_steps, warmup_steps + cosine_steps]
        )
        scheduler.load_state_dict(torch.load(f"{save_dir}/gpt-{load_timestamp}.sch.pt"))

        with open(f"{save_dir}/gpt-{load_timestamp}.json", "r", encoding="utf-8") as f:
            load_meta = json.load(f)

        start_batch_idx = load_meta["iteration"] + 1

    model = torch.compile(model)

    dataset_name = "fineweb-edu"
    loader = DataLoader(
        TokenizedBatchDataset(
            dataset=dataset_name,
            seq_len=seq_len,
            batch_size=batch_size,
            start_batch_idx=start_batch_idx,
            max_batches=n_batches - start_batch_idx,
        ),
        batch_size=None,
        num_workers=1,
        prefetch_factor=2,
        persistent_workers=True,
        pin_memory=True,
    )

    if do_train:
        model.train()
    else:
        model.eval()

    sum_loss = 0
    sum_tokens = 0

    ema_loss_numer = 0
    ema_loss_denom = 0
    ema_loss_alpha = 0.1

    ppl_plot_x = []
    ppl_plot_y = []

    pbar = tqdm(loader, total=n_batches, initial=start_batch_idx, mininterval=0, ncols=80)

    for batch in pbar:

        batch = batch.to(device)
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(inputs)
            loss = cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                reduction="sum"
            )
            if do_train:
                loss.backward()

        if do_train:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        sum_tokens += targets.numel()
        sum_loss += loss.item()

        if (pbar.n + 1) % stat_interval == 0:
            avg_loss = sum_loss / sum_tokens
            ppl = math.exp(avg_loss)
            sum_tokens = 0
            sum_loss = 0
            lr = optimizer.param_groups[0]["lr"]
            if do_train:
                print(f"Learning rate: {lr:.6f}, Perplexity: {ppl:.2f}")
            else:
                print(f"Learning rate: 0 (eval), Perplexity: {ppl:.2f}")

            ema_loss_numer = avg_loss * ema_loss_alpha + ema_loss_numer * (1 - ema_loss_alpha)
            ema_loss_denom = ema_loss_alpha + ema_loss_denom * (1 - ema_loss_alpha)
            ppl_plot_x.append(pbar.n)
            ppl_plot_y.append(ppl)

        if (pbar.n + 1) % save_interval == 0:
            save_checkpoint(
                meta={
                    "d_model": model.d_model,
                    "n_layers": model.n_layers,
                    "n_heads": model.n_heads,
                    "seq_len": seq_len,
                    "seed": global_seed,
                    "batch_size": batch_size,
                    "max_lr": max_lr,
                    "min_lr": min_lr,
                    "weight_decay": weight_decay,
                    "dataset": dataset_name,
                    "n_batches": n_batches,
                    "finished": False,
                    "iteration": pbar.n,
                    "avg_perplexity": math.exp(ema_loss_numer / ema_loss_denom)
                },
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                ppl_plot_x=ppl_plot_x,
                ppl_plot_y=ppl_plot_y,
                show_plt=False
            )

    save_checkpoint(
        meta={
            "d_model": model.d_model,
            "n_layers": model.n_layers,
            "n_heads": model.n_heads,
            "seq_len": seq_len,
            "seed": global_seed,
            "batch_size": batch_size,
            "max_lr": max_lr,
            "min_lr": min_lr,
            "weight_decay": weight_decay,
            "dataset": dataset_name,
            "n_batches": n_batches,
            "finished": True,
            "iteration": pbar.n - 1,
            "avg_perplexity": math.exp(ema_loss_numer / ema_loss_denom)
        },
        model=model,
        ppl_plot_x=ppl_plot_x,
        ppl_plot_y=ppl_plot_y,
        show_plt=False
    )
