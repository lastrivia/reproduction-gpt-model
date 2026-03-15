# from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.hub.api import HubApi
from tqdm import tqdm

repo_id = "epfml/FineWeb-HQ"
local_save = "./downloaded"
with open("../msc_token", "r") as f:
    msc_token = f.read().strip()

api = HubApi(token=msc_token)
repo_dirs = api.get_dataset_files(
    repo_id=repo_id,
    root_path='data',
    recursive=False
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
    get_files = [i["Path"] for i in get_files]
    repo_files.extend(get_files)

import json
with open("index.json", "w") as f:
    json.dump(repo_files, f)

# import random
# random.seed(42)
# sampled_indices = sorted(random.sample(range(1, 1001), 457))
# files = [f"chunk/pubmed23n{i:04d}.jsonl" for i in sampled_indices]
#
# # print(files)
# snapshot_download(
#     repo_id=repo_id,
#     repo_type="dataset",
#     allow_patterns=files,
#     local_dir=local_save,
#     token=msc_token,
#     max_workers=32
# )