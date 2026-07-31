import csv
from datetime import datetime
import json
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from utils.plot import plot_training_curve


LOG_FIELDS = ("step", "loss", "ppl", "lr")
CURVE_WINDOW_TOKENS = 4194304

def save_checkpoint(
        *,
        save_dir: str | Path,
        meta: dict,
        log_rows: Sequence[Mapping[str, int | float]],
        model: nn.Module,
        optimizer: Optimizer | None = None,
        scheduler: LRScheduler | None = None,
        finished: bool = False,
        error: bool = False,
        diagnosis: dict | None = None,
        max_backup: int = 2,
):
    """
    Save a directory checkpoint for the new training pipeline.
    """

    save_dir = Path(save_dir)
    finished_dir = save_dir / "finished"
    if finished_dir.exists():
        warnings.warn(
            f"checkpoint is already finished; skip saving: {finished_dir}",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    save_dir.mkdir(parents=True, exist_ok=True)

    # ================================
    # write checkpoint to temp dir
    # ================================

    tmp_dir = save_dir / "tmp"
    if tmp_dir.exists():
        if tmp_dir.is_dir():
            shutil.rmtree(tmp_dir)
        else:
            tmp_dir.unlink()
    tmp_dir.mkdir()

    with open(tmp_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)

    if error:
        with open(tmp_dir / "diagnosis.json", "w", encoding="utf-8") as f:
            json.dump(diagnosis, f, indent=4)

    with open(tmp_dir / "log.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in log_rows:
            writer.writerow({field: row[field] for field in LOG_FIELDS})

    curve_window = CURVE_WINDOW_TOKENS // meta["tokens_per_step"]
    if curve_window < 8:
        curve_window = 8

    plot_training_curve(
        step=[row["step"] for row in log_rows],
        log_ppl=[row["loss"] for row in log_rows],
        window=curve_window,
        save=tmp_dir / "curve.png"
    )

    torch.save(model.state_dict(), tmp_dir / "model.pt")
    if optimizer is not None:
        torch.save(optimizer.state_dict(), tmp_dir / "optimizer.pt")
    if scheduler is not None:
        torch.save(scheduler.state_dict(), tmp_dir / "scheduler.pt")

    # ================================
    # publish checkpoint
    # ================================

    if error:
        target = _error_checkpoint_dir(save_dir)
    else:
        _rotate_backups(save_dir=save_dir, max_backup=max_backup)
        target = save_dir / ("finished" if finished else "latest")

    if target.exists():
        shutil.rmtree(target)
    shutil.move(str(tmp_dir), str(target))


def _error_checkpoint_dir(save_dir: Path) -> Path:
    stem = f"error-{datetime.now().strftime('%m%d%H%M')}"
    target = save_dir / stem
    suffix = 1
    while target.exists():
        target = save_dir / f"{stem}-{suffix}"
        suffix += 1
    return target


def _rotate_backups(*, save_dir: Path, max_backup: int):
    latest = save_dir / "latest"
    if not latest.exists():
        return

    if max_backup <= 0:
        shutil.rmtree(latest)
        return

    overflow = save_dir / f"bak_{max_backup - 1}"
    if overflow.exists():
        shutil.rmtree(overflow)

    for i in range(max_backup - 2, -1, -1):
        src = save_dir / f"bak_{i}"
        if src.exists():
            dst = save_dir / f"bak_{i + 1}"
            if dst.exists():
                shutil.rmtree(dst)
            shutil.move(str(src), str(dst))

    shutil.move(str(latest), str(save_dir / "bak_0"))


@dataclass
class CheckpointState:
    optimizer: Optimizer | None
    scheduler: LRScheduler | None
    meta: dict | None
    log_rows: list[dict]
    checkpoint_dir: Path | None
    finished: bool
    resumed: bool


@dataclass
class CheckpointMeta:
    meta: dict | None
    checkpoint_dir: Path | None
    finished: bool


def load_checkpoint_meta(save_dir: str | Path) -> CheckpointMeta:
    save_dir = Path(save_dir)
    finished_dir = save_dir / "finished"
    latest_dir = save_dir / "latest"

    if finished_dir.exists():
        checkpoint_dir = finished_dir
        finished = True
    elif latest_dir.exists():
        checkpoint_dir = latest_dir
        finished = False
    else:
        return CheckpointMeta(meta=None, checkpoint_dir=None, finished=False)

    with open(checkpoint_dir / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    return CheckpointMeta(meta=meta, checkpoint_dir=checkpoint_dir, finished=finished)


def init_checkpoint(
        *,
        save_dir: str | Path,
        model: nn.Module,
        device: torch.device | str,
        build_optimizer: Callable[[nn.Module], Optimizer] | None = None,
        build_scheduler: Callable[[Optimizer], LRScheduler] | None = None,
) -> CheckpointState:

    checkpoint_meta = load_checkpoint_meta(save_dir)
    checkpoint_dir = checkpoint_meta.checkpoint_dir
    finished = checkpoint_meta.finished

    meta = checkpoint_meta.meta
    log_rows = []

    # ================================
    # load metadata and model
    # ================================

    if checkpoint_dir is not None:
        with open(checkpoint_dir / "log.csv", "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            log_rows = [
                {
                    "step": int(row["step"]),
                    "loss": float(row["loss"]),
                    "ppl": float(row["ppl"]),
                    "lr": float(row["lr"]),
                }
                for row in reader
            ]

        model.load_state_dict(torch.load(checkpoint_dir / "model.pt", map_location="cpu"))

    model.to(device)

    optimizer = build_optimizer(model) if build_optimizer is not None else None
    scheduler = build_scheduler(optimizer) if build_scheduler is not None else None

    # ================================
    # resume training status
    # ================================

    resumed = checkpoint_dir is not None and not finished
    if resumed:
        optimizer_path = checkpoint_dir / "optimizer.pt"
        scheduler_path = checkpoint_dir / "scheduler.pt"

        if optimizer is not None and optimizer_path.exists():
            optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu"))
            for state in optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(device)

        if scheduler is not None and scheduler_path.exists():
            scheduler.load_state_dict(torch.load(scheduler_path, map_location="cpu"))

    return CheckpointState(
        optimizer=optimizer,
        scheduler=scheduler,
        meta=meta,
        log_rows=log_rows,
        checkpoint_dir=checkpoint_dir,
        finished=finished,
        resumed=resumed,
    )
