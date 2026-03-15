# from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download
from modelscope.hub.snapshot_download import snapshot_download

repo_id = "OmniData/Pile-OpenWebText2"
allow_patterns="*.jsonl"
local_save = "./downloaded"
with open("../msc_token", "r") as f:
    msc_token = f.read().strip()

snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    allow_patterns=allow_patterns,
    local_dir=local_save,
    token=msc_token,
    max_workers=32
)