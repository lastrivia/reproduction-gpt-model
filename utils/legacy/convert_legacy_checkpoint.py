import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Mapping

import torch

from convert_classic_model import convert_classic_state_dict
from unwrap_compiled import unwrap_compiled_state_dict


LOG_FIELDS = ("step", "loss", "ppl", "lr")
FINEWEB_VOCAB_SIZE = 50280


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert fineweb-edu legacy checkpoints to finished/ checkpoint dirs."
    )
    parser.add_argument("source", type=Path, help="Legacy .pt, legacy run dir, or legacy root dir.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output dir for one checkpoint, or output root for multiple run dirs.",
    )
    parser.add_argument("--model-preset", help="Override inferred model_preset.")
    parser.add_argument("--dataset", help="Override dataset from legacy metadata.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing finished dir.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned conversions without writing files.")

    tanh_group = parser.add_mutually_exclusive_group()
    tanh_group.add_argument("--tanh", dest="tanh", action="store_true", help="Mark dynamic HC as tanh gated.")
    tanh_group.add_argument("--no-tanh", dest="tanh", action="store_false", help="Mark dynamic HC as non-tanh.")
    parser.set_defaults(tanh=None)

    return parser.parse_args()


def find_model_pt(source: Path) -> Path | None:
    if source.is_file():
        return source
    candidates = [
        path
        for path in source.glob("*.pt")
        if not path.name.endswith(".opt.pt") and not path.name.endswith(".sch.pt")
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def discover_sources(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if find_model_pt(source) is not None:
        return [source]
    return [
        child
        for child in sorted(source.iterdir())
        if child.is_dir() and find_model_pt(child) is not None
    ]


def default_output_dir(source: Path, output: Path | None, multiple: bool) -> Path:
    source_dir = source.parent if source.is_file() else source
    if output is not None:
        return output / source_dir.name if multiple else output
    if source_dir.parent.name == "legacy":
        return source_dir.parent.parent / source_dir.name
    return source_dir.with_name(f"{source_dir.name}-converted")


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_state_dict(state_dict: Mapping):
    unwrapped = unwrap_compiled_state_dict(state_dict)
    if unwrapped is not None:
        state_dict = unwrapped

    converted = convert_classic_state_dict(state_dict)
    if converted is not None:
        state_dict = converted

    return state_dict


def detect_arch(state_dict: Mapping) -> tuple[str, dict]:
    keys = set(state_dict)
    if any(key.startswith("decoders.") for key in keys):
        return "vanilla", {}

    if any(key.startswith("attn_blocks.") for key in keys):
        a = state_dict.get("attn_blocks.0.a")
        if a is None:
            raise KeyError("HC checkpoint is missing attn_blocks.0.a")
        expansion_rate = a.shape[0]
        dynamic = "attn_blocks.0.w_a" in keys
        return "hc", {
            "expansion_rate": expansion_rate,
            "dynamic": dynamic,
        }

    raise NotImplementedError("could not detect legacy architecture")


def infer_model_preset(meta: dict) -> str:
    n_layers = meta["n_layers"]
    d_model = meta["d_model"]
    seq_len = meta.get("seq_len", meta.get("max_len"))
    batch_size = meta.get("batch_size")

    if (n_layers, d_model, seq_len, batch_size) == (12, 768, 512, 8):
        return "small-classic"
    if (n_layers, d_model, seq_len, batch_size) == (6, 512, 512, 16):
        return "smallest-classic"

    raise ValueError(
        "could not infer model_preset; pass --model-preset "
        f"for n_layers={n_layers}, d_model={d_model}, seq_len={seq_len}, batch_size={batch_size}"
    )


def build_new_meta(
        *,
        legacy_meta: dict,
        state_dict: Mapping,
        source_pt: Path,
        source_json: Path,
        residual_arch: str,
        arch_params: dict,
        model_preset: str,
        dataset: str,
):
    seq_len = legacy_meta.get("seq_len", legacy_meta.get("max_len"))
    batch_size = legacy_meta["batch_size"]
    n_batches = legacy_meta["n_batches"]
    step = legacy_meta["iteration"]
    avg_perplexity = legacy_meta.get("avg_perplexity")
    avg_loss = math.log(avg_perplexity) if avg_perplexity and avg_perplexity > 0 else None

    meta = {
        "model_preset": model_preset,
        "residual_arch": residual_arch,
        "compatible": True,
        "n_layers": legacy_meta["n_layers"],
        "d_model": legacy_meta["d_model"],
        "n_heads": legacy_meta["n_heads"],
        "norm": "LayerNorm",
        "seq_len": seq_len,
        "seed": legacy_meta.get("seed"),
        "micro_batch_size": batch_size,
        "tokens_per_step": batch_size * seq_len,
        "grad_accum_steps": 1,
        "max_lr": legacy_meta.get("max_lr"),
        "min_lr": legacy_meta.get("min_lr"),
        "warmup_ratio": 0.01,
        "cosine_ratio": 0.95,
        "weight_decay": legacy_meta.get("weight_decay", 0.01),
        "grad_clip_norm": None,
        "dataset": dataset,
        "n_steps": n_batches,
        "training_tokens": n_batches * batch_size * seq_len,
        "stat_interval_s": None,
        "save_interval_s": None,
        "avg_count": None,
        "avg_loss": avg_loss,
        "avg_perplexity": avg_perplexity,
        "finished": True,
        "step": step,
        "vocab_size": state_dict["embedding.weight"].shape[0],
        "converted_from": {
            "checkpoint": str(source_pt),
            "metadata": str(source_json),
        },
    }
    if arch_params:
        meta["arch_params"] = arch_params
    return meta


def write_log(path: Path, meta: dict):
    avg_ppl = meta.get("avg_perplexity")
    avg_loss = meta.get("avg_loss")
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        writer.writeheader()
        if avg_ppl is not None and avg_loss is not None:
            writer.writerow({
                "step": meta["step"],
                "loss": avg_loss,
                "ppl": avg_ppl,
                "lr": meta.get("min_lr") or 0.0,
            })


def convert_one(source: Path, output_dir: Path, args) -> bool:
    source_pt = find_model_pt(source)
    if source_pt is None:
        print(f"[skip] no model pt found: {source}")
        return False

    source_json = source_pt.with_suffix(".json")
    if not source_json.exists():
        raise FileNotFoundError(f"missing legacy metadata: {source_json}")

    legacy_meta = load_json(source_json)
    state_dict = torch.load(source_pt, map_location="cpu")
    if not isinstance(state_dict, Mapping):
        raise TypeError(f"expected a state_dict: {source_pt}")
    state_dict = normalize_state_dict(state_dict)

    vocab_size = state_dict["embedding.weight"].shape[0]
    dataset = args.dataset or legacy_meta.get("dataset")
    if dataset != "fineweb-edu" or vocab_size != FINEWEB_VOCAB_SIZE:
        print(f"[skip] unsupported legacy checkpoint: {source_pt}")
        print(f"       dataset={dataset!r}, vocab_size={vocab_size}")
        return False

    residual_arch, arch_params = detect_arch(state_dict)
    if residual_arch == "hc":
        if arch_params["dynamic"]:
            source_name = (source.parent if source.is_file() else source).name.lower()
            arch_params["tanh"] = args.tanh if args.tanh is not None else "dhc" in source_name
        else:
            arch_params["tanh"] = False

    model_preset = args.model_preset or infer_model_preset(legacy_meta)
    meta = build_new_meta(
        legacy_meta=legacy_meta,
        state_dict=state_dict,
        source_pt=source_pt,
        source_json=source_json,
        residual_arch=residual_arch,
        arch_params=arch_params,
        model_preset=model_preset,
        dataset=dataset,
    )

    finished_dir = output_dir / "finished"
    print(f"[convert] {source_pt} -> {finished_dir}")
    print(f"          arch={residual_arch}, preset={model_preset}, step={meta['step']}")
    if args.dry_run:
        return True

    if finished_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"finished dir already exists: {finished_dir}")
        shutil.rmtree(finished_dir)
    finished_dir.mkdir(parents=True)

    torch.save(state_dict, finished_dir / "model.pt")
    with open(finished_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)
    write_log(finished_dir / "log.csv", meta)

    source_curve = source_pt.with_name(f"{source_pt.stem}-curve.png")
    if source_curve.exists():
        shutil.copy2(source_curve, finished_dir / "curve.png")

    return True


def main():
    args = parse_args()
    sources = discover_sources(args.source)
    if not sources:
        raise FileNotFoundError(f"no legacy checkpoints found under {args.source}")

    multiple = len(sources) > 1
    converted = 0
    for source in sources:
        output_dir = default_output_dir(source, args.output, multiple)
        if convert_one(source, output_dir, args):
            converted += 1

    if converted == 0:
        raise RuntimeError("no checkpoints converted")
    print(f"Converted {converted} checkpoint(s).")


if __name__ == "__main__":
    main()
