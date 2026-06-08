# from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download

import os
from modelscope.hub.api import HubApi
from tqdm import tqdm
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

repo_id = "AI-ModelScope/fineweb-edu"
local_save = BASE_DIR / "downloaded"

api = HubApi(token=os.environ["MODELSCOPE_TOKEN"])
repo_dirs = api.get_dataset_files(
    repo_id=repo_id,
    root_path='data',
    recursive=False,
    page_size=4096
)
repo_dirs = [i["Path"] for i in repo_dirs]

repo_files = []
for repo_dir in tqdm(repo_dirs, desc="Loading dirs"):
    get_files = api.get_dataset_files(
        repo_id=repo_id,
        root_path=repo_dir,
        recursive=False,
        page_size=4096
    )
    get_files = [
        {
            "Path": i["Path"],
            "Size": i["Size"]
        }
        for i in get_files
    ]
    repo_files.extend(get_files)

import json
with open(BASE_DIR / "index.json", "w", encoding="utf-8") as f:
    json.dump(repo_files, f, indent=4)
