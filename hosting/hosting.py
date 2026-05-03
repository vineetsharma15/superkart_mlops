from huggingface_hub import HfApi, create_repo
import os

# Environment variables
HF_TOKEN = os.getenv("HF_TOKEN")
SPACE_REPO_ID = os.getenv("SPACE_REPO_ID", "vsharma15/superkart-sales-forecasting-app")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set.")

# Initialize API
api = HfApi(token=HF_TOKEN)

# Step 1: Create Hugging Face Space if it does not exist
create_repo(
    repo_id=SPACE_REPO_ID,
    repo_type="space",
    space_sdk="docker",
    token=HF_TOKEN,
    exist_ok=True
)

print(f"Space ready: {SPACE_REPO_ID}")

# Step 2: Upload deployment folder
api.upload_folder(
    folder_path="deployment",
    repo_id=SPACE_REPO_ID,
    repo_type="space",
    path_in_repo="",
    token=HF_TOKEN
)

print("Deployment files uploaded successfully to Hugging Face Space.")
