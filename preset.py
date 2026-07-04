from pathlib import Path
from typing import Any

import yaml


PRESET_FILE = Path(__file__).resolve().with_name("preset.yaml")


def load_presets(path: str | Path = PRESET_FILE) -> dict[str, dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    resolved = {}
    for name in raw:
        resolved[name] = _resolve_preset(name, raw, resolved, stack=[])
    return {name: dict(config) for name, config in resolved.items()}


def load_preset(name: str, path: str | Path = PRESET_FILE) -> dict[str, Any]:
    presets = load_presets(path)
    if name not in presets:
        raise KeyError(f"unknown preset: {name}")
    return dict(presets[name])


def _resolve_preset(
        name: str,
        raw: dict[str, dict[str, Any]],
        resolved: dict[str, dict[str, Any]],
        stack: list[str],
) -> dict[str, Any]:
    if name in resolved:
        return resolved[name]
    if name in stack:
        cycle = " -> ".join(stack + [name])
        raise ValueError(f"cyclic preset inheritance: {cycle}")
    if name not in raw:
        raise KeyError(f"unknown preset: {name}")

    config = raw[name]
    base_name = config.get("derived")
    if base_name is None:
        merged = {}
    else:
        merged = dict(_resolve_preset(base_name, raw, resolved, stack + [name]))

    for key, value in config.items():
        if key != "derived":
            merged[key] = value

    resolved[name] = merged
    return merged
