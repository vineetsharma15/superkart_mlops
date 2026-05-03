from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
import os

repo_id = "vsharma15/superkart_mlops"
repo_type = "dataset"
folder_path = "data"

hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN environment variable is not set.")

api = HfApi(token=hf_token)

# Check whether the dataset repo already exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset repo '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Dataset repo '{repo_id}' not found. Creating it...")
    create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=False,
        token=hf_token
    )
    print(f"Dataset repo '{repo_id}' created.")

# Upload local folder contents to the dataset repo
api.upload_folder(
    folder_path=folder_path,
    repo_id=repo_id,
    repo_type=repo_type,
)
print(f"Uploaded contents of '{folder_path}' to https://huggingface.co/datasets/{repo_id}")
print("Raw data registered successfully on Hugging Face Dataset Hub.")
