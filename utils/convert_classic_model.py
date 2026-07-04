import argparse
from collections import OrderedDict
from pathlib import Path
from typing import Mapping

import torch


RENAMES = (
    (".layer_norm.", ".dhc_norm."),
    ("final_ln.", "final_norm."),
)


def default_output_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_converted{path.suffix}")


def convert_classic_state_dict(state_dict: Mapping):
    keys = list(state_dict.keys())

    if not keys:
        return OrderedDict()

    if not all(isinstance(key, str) for key in keys):
        raise TypeError("Expected a model state_dict with string keys.")

    converted = OrderedDict()
    changed = False
    for key, value in state_dict.items():
        new_key = key
        for old, new in RENAMES:
            new_key = new_key.replace(old, new)
        if new_key != key:
            changed = True
        if new_key in converted:
            raise ValueError(f"Key collision while converting {key!r} to {new_key!r}.")
        converted[new_key] = value

    return converted if changed else None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert legacy Transformer model state_dict keys to the current names."
    )
    parser.add_argument("checkpoint", type=Path, help="Path to the model .pt file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output path. Defaults to *_converted.pt next to the input.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    output_path = args.output or default_output_path(checkpoint_path)

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(state_dict, Mapping):
        raise TypeError("Expected the checkpoint to be a model state_dict.")

    converted = convert_classic_state_dict(state_dict)
    if converted is None:
        print("No classic key names detected; no file written.")
        return 0

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")

    torch.save(converted, output_path)
    print(f"Saved converted checkpoint to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
