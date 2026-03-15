from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download

repo_id = "allenai/c4"
# allow_patterns=
local_save = "./downloaded"
with open("../hf_token", "r") as f:
    hf_token = f.read().strip()

import random
random.seed(42)
sampled_indices = sorted(random.sample(range(0, 1024), 154))
files = [f"en/c4-train.{i:05d}-of-01024.json.gz" for i in sampled_indices]

snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    allow_patterns=files,
    local_dir=local_save,
    token=hf_token,
    max_workers=32
)