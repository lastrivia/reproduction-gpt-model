# from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download
from modelscope.hub.snapshot_download import snapshot_download

repo_id = "MedRAG/pubmed"
# allow_patterns="chunk/*"
local_save = "./downloaded"
with open("../msc_token", "r") as f:
    msc_token = f.read().strip()

# import random
# random.seed(42)
# sampled_indices = sorted(random.sample(range(1, 1001), 457))
# files = [f"chunk/pubmed23n{i:04d}.jsonl" for i in sampled_indices]

# print(files)
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    # allow_patterns=files,
    local_dir=local_save,
    token=msc_token,
    max_workers=32
)