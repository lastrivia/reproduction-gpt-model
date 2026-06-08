from modelscope.hub.snapshot_download import snapshot_download
from collections import defaultdict
import os
from pathlib import Path
import random
import re
import json

BASE_DIR = Path(__file__).resolve().parent

repo_id = "AI-ModelScope/fineweb-edu"
local_save = BASE_DIR / "downloaded"
MAX_BYTES_PER_YEAR = 5_000_000_000

with open(BASE_DIR / "index.json", "r", encoding="utf-8") as f:
    repo_files = json.load(f)

files_by_year = defaultdict(list)
for repo_file in repo_files:
    match = re.search(r"CC-MAIN-(\d{4})-", repo_file["Path"])
    if not match:
        continue
    files_by_year[match.group(1)].append(repo_file)

random.seed(42)
files = []
for year in sorted(files_by_year):
    candidates = files_by_year[year][:]
    random.shuffle(candidates)

    total_size = 0
    selected = []
    for repo_file in candidates:
        size = int(repo_file["Size"])
        if total_size + size > MAX_BYTES_PER_YEAR:
            continue
        selected.append(repo_file["Path"])
        total_size += size

    files.extend(selected)
    print(f"{year}: {len(selected)} files, {total_size / 1e9:.2f} GB")

print(f"Total: {len(files)} files")

snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    allow_patterns=files,
    local_dir=local_save,
    token=os.environ["MODELSCOPE_TOKEN"],
    max_workers=32
)
