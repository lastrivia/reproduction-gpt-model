import json
from pathlib import Path

import numpy as np
import zstandard as zstd


_DATA_DIR = Path(__file__).resolve().parent


def _file_record(file: Path, tokens: int | None = None) -> dict:
    stat = file.stat()
    record = {
        "path": file.name,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if tokens is not None:
        record["tokens"] = tokens
    return record


def _manifest_matches(manifest: list[dict], files: list[Path]) -> bool:
    if len(manifest) != len(files):
        return False
    for item, file in zip(manifest, files):
        record = _file_record(file)
        if item.get("path") != record["path"]:
            return False
        if item.get("size") != record["size"]:
            return False
        if item.get("mtime_ns") != record["mtime_ns"]:
            return False
        if "tokens" not in item:
            return False
    return True


def _count_tokens(file: Path, rbuf_size: int, print_file_ops: bool) -> int:
    if print_file_ops:
        print(f"[Manifest] Counting {file}")

    n_bytes = 0
    dctx = zstd.ZstdDecompressor()
    with open(file, "rb") as f:
        with dctx.stream_reader(f) as reader:
            while True:
                chunk = reader.read(rbuf_size)
                if not chunk:
                    break
                n_bytes += len(chunk)

    if n_bytes % np.dtype(np.uint32).itemsize != 0:
        raise ValueError(f"tokenized chunk is not uint32-aligned: {file}")
    return n_bytes // np.dtype(np.uint32).itemsize


def load_manifest(
        dataset: str,
        rbuf_size: int = 1 << 20,
        print_file_ops: bool = False,
        data_dir: str | Path = _DATA_DIR
) -> list[dict]:
    """Return tokenized chunk metadata for data/{dataset}/tokenized.

    The returned manifest is a list ordered by chunk file name. Each item has:
    - path: chunk file name relative to the tokenized directory
    - size: compressed file size in bytes
    - mtime_ns: file modification time in nanoseconds
    - tokens: number of uint32 tokens in the decompressed chunk
    """
    tokenized_dir = Path(data_dir) / dataset / "tokenized"
    files = sorted(tokenized_dir.glob("chunk-*.bin.zst"))
    if len(files) == 0:
        raise FileNotFoundError(f"no tokenized chunks found under {tokenized_dir}")

    manifest_file = tokenized_dir / "manifest.json"

    build_manifest = False
    if manifest_file.exists():
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        if not _manifest_matches(manifest, files):
            build_manifest = True
    else:
        build_manifest = True

    if build_manifest:
        manifest = []
        for file in files:
            tokens = _count_tokens(file, rbuf_size, print_file_ops)
            manifest.append(_file_record(file, tokens=tokens))

        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=4)

    return manifest
