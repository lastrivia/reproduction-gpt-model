# from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.hub.api import HubApi

repo_id = "epfml/FineWeb-HQ"
local_save = "./downloaded"
with open("../msc_token", "r") as f:
    msc_token = f.read().strip()

import json
with open("index.json", "r") as f:
    repo_files = json.load(f)

import random
random.seed(42)
files = sorted(random.sample(repo_files, 128))
print(files)

snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    allow_patterns=files,
    local_dir=local_save,
    token=msc_token,
    max_workers=32
)