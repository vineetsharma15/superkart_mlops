
# ----------------------------
# Imports
# ----------------------------
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

# ----------------------------
# Hugging Face API
# ----------------------------
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN environment variable is not set.")

api = HfApi(token=hf_token)
DATASET_REPO_ID = os.getenv("HF_DATASET_REPO_ID", "vsharma15/superkart_mlops")
DATASET_PATH = f"hf://datasets/{DATASET_REPO_ID}/SuperKart.csv"

# ----------------------------
# Load dataset from Hugging Face
# ----------------------------
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully:", df.shape)

# ----------------------------
# Data cleaning
# ----------------------------
target = "Product_Store_Sales_Total"

# Remove unnecessary identifier columns
unnecessary_cols = ["Product_Id"]
df = df.drop(columns=unnecessary_cols, errors="ignore")

# Standardize categorical values where required
df["Product_Sugar_Content"] = df["Product_Sugar_Content"].replace({
    "low sugar": "Low Sugar",
    "LF": "Low Sugar",
    "reg": "Regular"
})

# Fill missing numerical and categorical values
num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# ----------------------------
# Train-test split
# ----------------------------
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

os.makedirs("data", exist_ok=True)
train_df.to_csv("data/train.csv", index=False)
test_df.to_csv("data/test.csv", index=False)

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

# ----------------------------
# Upload processed files back to Hugging Face Dataset Hub
# ----------------------------
api.upload_file(
    path_or_fileobj="data/train.csv",
    path_in_repo="train.csv",
    repo_id=DATASET_REPO_ID,
    repo_type="dataset",
    token=hf_token
)

api.upload_file(
    path_or_fileobj="data/test.csv",
    path_in_repo="test.csv",
    repo_id=DATASET_REPO_ID,
    repo_type="dataset",
    token=hf_token
)

print("Train and test datasets uploaded successfully to Hugging Face Dataset Hub.")
