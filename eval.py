import argparse
import json
import math
from pathlib import Path
from typing import Mapping

import torch
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import TokenizedBatchDataset
from transformer.hc_transformer import HCTransformer
from transformer.mhc_transformer import MHCTransformer
from transformer.transformer import Transformer


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint and print perplexity.")
    parser.add_argument("checkpoint", type=Path, help="Path to model.pt, a checkpoint dir, or a save dir.")
    parser.add_argument("-s", "--seq-len", required=True, type=int)
    parser.add_argument("-b", "--batch-size", required=True, type=int)
    parser.add_argument("-i", "--start-batch-idx", required=True, type=int)
    parser.add_argument("-n", "--n-batches", required=True, type=int)
    parser.add_argument("-c", "--cuda", type=int, default=0)
    return parser.parse_args()


def resolve_model_path(path: Path) -> Path:
    if path.is_file():
        return path

    candidates = (
        path / "model.pt",
        path / "finished" / "model.pt",
        path / "latest" / "model.pt",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"could not find model.pt under {path}")


def load_meta(model_path: Path) -> dict:
    meta_path = model_path.parent / "meta.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_state_dict_vocab_size(state_dict: Mapping) -> int:
    return state_dict["embedding.weight"].shape[0]


def build_model(meta: dict, state_dict: Mapping):
    residual_arch = meta["residual_arch"]
    common_kwargs = {
        "n_layers": meta["n_layers"],
        "d_model": meta["d_model"],
        "n_heads": meta["n_heads"],
        "vocab_size": get_state_dict_vocab_size(state_dict),
        "norm": meta.get("norm", "LayerNorm"),
        "dropout": 0.1,
    }

    if residual_arch == "vanilla":
        return Transformer(**common_kwargs)
    if residual_arch == "hc":
        return HCTransformer(**common_kwargs, **meta.get("arch_params", {}))
    if residual_arch == "mhc":
        arch_params = dict(meta.get("arch_params", {}))
        if meta.get("use_legacy_mhc", False):
            arch_params.setdefault("scale_norm", False)
        else:
            arch_params.setdefault("scale_norm", meta.get("scale_norm", True))
        return MHCTransformer(**common_kwargs, **arch_params)

    raise NotImplementedError(f"unknown residual_arch: {residual_arch}")


def main():
    args = parse_args()

    model_path = resolve_model_path(args.checkpoint)
    meta = load_meta(model_path)
    state_dict = torch.load(model_path, map_location="cpu")

    device = torch.device(f"cuda:{args.cuda}")
    model = build_model(meta, state_dict)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    loader = DataLoader(
        TokenizedBatchDataset(
            dataset=meta["dataset"],
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            start_batch_idx=args.start_batch_idx,
            max_batches=args.n_batches,
        ),
        batch_size=None,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    total_loss = 0.0
    total_tokens = 0
    with torch.inference_mode():
        for batch in tqdm(loader, total=len(loader), ncols=80):
            batch = batch.to(device, non_blocking=True)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(inputs)
                loss = cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    targets.reshape(-1),
                    reduction="sum",
                )

            total_loss += loss.item()
            total_tokens += targets.numel()

    avg_loss = total_loss / total_tokens
    print(f"loss: {avg_loss:.6f}")
    print(f"ppl: {math.exp(avg_loss):.6f}")
    print(f"tokens: {total_tokens}")
    print(f"batches: {len(loader)}")


if __name__ == "__main__":
    main()
