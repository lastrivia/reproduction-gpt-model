import math
import time
import os
import numpy as np
import glob
import json
import random
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from tokenizers.decoders import ByteLevel
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, ConstantLR
from torch.nn.functional import cross_entropy
from tqdm import tqdm

from transformer.transformer import Transformer
from dataset import ZstdTokenStreamDataset


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


save_dir = "weight"


def save_checkpoint(meta, model, optimizer=None, scheduler=None):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = time.strftime("%m%d-%H%M")
    with open(f"weight/gpt-{timestamp}.json", "w") as f:
        json.dump(meta, f)
    torch.save(model.state_dict(), f"weight/gpt-{timestamp}.pt")
    if optimizer is not None:
        torch.save(optimizer.state_dict(), f"weight/gpt-{timestamp}.opt.pt")
    if scheduler is not None:
        torch.save(scheduler.state_dict(), f"weight/gpt-{timestamp}.sch.pt")
    print(f"Checkpoint {timestamp} saved; Avg perplexity: {meta['avg_perplexity']}")


def preset(name): # n_layers, d_model, n_heads, batch_size, max_lr, min_lr
    match name:
        case "smallest":    # 50M params,  9G VRAM
            return 6, 512, 8, 16, 5e-4, 5e-5
        case "small":       # 151M params, 8G VRAM
            return 12, 768, 12, 8, 2e-4, 2e-5
        case "medium":      # 353M params, 9G VRAM
            return 18, 1024, 16, 4, 1e-4, 1e-5
    raise NotImplementedError

if __name__ == "__main__":
    set_seed(42)

    tokenizer = Tokenizer.from_file("tokenizer/trained.json")
    tokenizer.decoder = ByteLevel()
    vocab_size = tokenizer.get_vocab_size()
    print("Vocab size:", vocab_size)

    training_set = glob.glob("data/*/chunk-*-tokenized.bin.zst")
    random.shuffle(training_set)

    training_set_start = 0
    training_set_end = 17  # suggested: Params (M) / 3
    training_set = training_set[training_set_start:training_set_end]
    print(training_set)

    n_layers, d_model, n_heads, batch_size, max_lr, min_lr = preset("smallest")

    model = Transformer(
        n_layers=n_layers,
        d_model=d_model,
        n_heads=n_heads,
        vocab_size=vocab_size,
        max_len=512
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {total_params:,}")

    device = torch.device("cuda:0")

    # total ~ 80B tokens
    est_tokens_per_file = 60000000
    est_tokens = len(training_set) * est_tokens_per_file
    est_total_steps = est_tokens // batch_size // model.max_len
    print(f"Training tokens (est): {est_tokens:,}")

    warmup_steps = int(est_total_steps * 0.01)
    cosine_steps = int(est_total_steps * 0.95)

    loader = DataLoader(
        dataset=ZstdTokenStreamDataset(
            files=training_set,
            seq_len=model.max_len,
            parallel_files=16
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    optimizer = AdamW(model.parameters(), lr=max_lr)
    scheduler = SequentialLR(
        optimizer=optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=1e-5, end_factor=1.0, total_iters=warmup_steps),
            CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=min_lr),
            ConstantLR(optimizer, factor=min_lr / max_lr)
        ],
        milestones=[warmup_steps, cosine_steps]
    )

    model.to(device)
    model.train()

    sum_loss = 0
    sum_tokens = 0
    stat_interval = 100
    save_interval = 10000

    ema_loss_numer = 0
    ema_loss_denom = 0
    ema_loss_alpha = 0.1

    pbar = tqdm(loader, total=est_total_steps, mininterval=0)
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
            loss.backward()

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        sum_tokens += targets.numel()
        sum_loss += loss.item()

        if pbar.n and pbar.n % stat_interval == 0:
            avg_loss = sum_loss / sum_tokens
            ppl = math.exp(avg_loss)
            sum_tokens = 0
            sum_loss = 0
            lr = optimizer.param_groups[0]["lr"]
            print(f"Learning rate: {lr:.6f}, Perplexity: {ppl:.2f}")

            ema_loss_numer = avg_loss * ema_loss_alpha + ema_loss_numer * (1 - ema_loss_alpha)
            ema_loss_denom = ema_loss_alpha + ema_loss_denom * (1 - ema_loss_alpha)

        if pbar.n and pbar.n % save_interval == 0:
            save_checkpoint(
                meta={
                    "d_model": model.d_model,
                    "n_layers": model.n_layers,
                    "n_heads": model.n_heads,
                    "max_len": model.max_len,
                    "batch_size": batch_size,
                    "max_lr": max_lr,
                    "min_lr": min_lr,
                    "training_set_start": training_set_start,
                    "training_set_end": training_set_end,
                    "finished": False,
                    "iteration": pbar.n,
                    "avg_perplexity": math.exp(ema_loss_numer / ema_loss_denom)
                },
                model=model,
                optimizer=optimizer,
                scheduler=scheduler
            )

    save_checkpoint(
        meta={
            "training_set_start": training_set_start,
            "training_set_end": training_set_end,
            "finished": True,
            "iteration": pbar.n,
            "avg_perplexity": math.exp(ema_loss_numer / ema_loss_denom)
        },
        model=model
    )
