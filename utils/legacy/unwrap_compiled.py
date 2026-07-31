import argparse
from collections import OrderedDict
from pathlib import Path
from typing import Mapping

import torch


COMPILED_PREFIX = "_orig_mod."


def default_output_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_unwrapped{path.suffix}")


def unwrap_compiled_state_dict(state_dict: Mapping):
    keys = list(state_dict.keys())

    if not keys:
        return None

    if not all(isinstance(key, str) for key in keys):
        raise TypeError("Expected a model state_dict with string keys.")

    prefixed = [key.startswith(COMPILED_PREFIX) for key in keys]
    if not any(prefixed):
        return None
    if not all(prefixed):
        raise ValueError("Found a mix of compiled and non-compiled state_dict keys.")

    return OrderedDict(
        (key[len(COMPILED_PREFIX):], value)
        for key, value in state_dict.items()
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert a torch.compile-generated model state_dict to normal keys."
    )
    parser.add_argument("checkpoint", type=Path, help="Path to the .pt file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output path. Defaults to x_unwrapped.pt next to x.pt.",
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

    unwrapped = unwrap_compiled_state_dict(state_dict)
    if unwrapped is None:
        print("No compiled-model prefix detected; no file written.")
        return 0

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")

    torch.save(unwrapped, output_path)
    print(f"Saved unwrapped checkpoint to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
