from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download

repo_id = "rojagtap/bookcorpus"
allow_patterns="*.txt"
local_save = "./downloaded"
with open("../hf_token", "r") as f:
    hf_token = f.read().strip()

snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    allow_patterns=allow_patterns,
    local_dir=local_save,
    token=hf_token,
    max_workers=32
)