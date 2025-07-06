import os
from huggingface_hub import snapshot_download

repo_id = "alexzyqi/VLN-Ego-making"
local_data_dir = "data"


print(f"Starting download for: {repo_id}")

os.makedirs(local_data_dir, exist_ok=True)

snapshot_download(
    repo_id=repo_id,
    local_dir=os.path.join(local_data_dir, repo_id.split('/')[-1]), # Save in a subfolder
    repo_type="dataset",
    resume_download=True
)

print(f"Download complete for {repo_id}.")

