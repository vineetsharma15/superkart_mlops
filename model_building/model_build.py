import pandas as pd
import numpy as np
import os
import joblib
import mlflow

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# ----------------------------
# MLflow setup
# ----------------------------
mlflow.set_experiment("SuperKart-Sales-Forecasting-Experiment")

# ----------------------------
# Hugging Face setup
# ----------------------------
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN environment variable is not set.")

api = HfApi(token=hf_token)

DATASET_REPO_ID = os.getenv("HF_DATASET_REPO_ID", "vsharma15/superkart_mlops")
MODEL_REPO_ID = os.getenv("HF_MODEL_REPO_ID", "vsharma15/superkart_mlops_model")

train_path = f"hf://datasets/{DATASET_REPO_ID}/train.csv"
test_path = f"hf://datasets/{DATASET_REPO_ID}/test.csv"

# ----------------------------
# Load train and test data from Hugging Face Dataset Hub
# ----------------------------
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print("Train data shape:", train_df.shape)
print("Test data shape:", test_df.shape)

# ----------------------------
# Define target and features
# ----------------------------
target = "Product_Store_Sales_Total"

X_train = train_df.drop(columns=[target])
y_train = train_df[target]
X_test = test_df.drop(columns=[target])
y_test = test_df[target]

numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()

print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)

# ----------------------------
# Preprocessing pipeline
# ----------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# ----------------------------
# Define models and hyperparameter grids
# ----------------------------
models = {
    "RandomForest": (
        RandomForestRegressor(random_state=42, n_jobs=-1),
        {
            "model__n_estimators": [100, 200],
            "model__max_depth": [10, 20, None],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2]
        }
    ),
    "XGBoost": (
        XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1
        ),
        {
            "model__n_estimators": [100, 200],
            "model__max_depth": [3, 5, 7],
            "model__learning_rate": [0.05, 0.1],
            "model__subsample": [0.8, 1.0],
            "model__colsample_bytree": [0.8, 1.0]
        }
    )
}

best_model = None
best_model_name = None
best_rmse = np.inf
results = []

# ----------------------------
# Tune models using GridSearchCV and evaluate performance
# ----------------------------
for model_name, (model, params) in models.items():
    print("\n" + "=" * 80)
    print(f"Training and tuning model: {model_name}")
    print("=" * 80)

    with mlflow.start_run(run_name=model_name):
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        grid = GridSearchCV(
            estimator=pipe,
            param_grid=params,
            cv=3,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            verbose=1
        )

        grid.fit(X_train, y_train)
        preds = grid.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        mlflow.log_param("model_name", model_name)
        mlflow.log_params(grid.best_params_)
        mlflow.log_metrics({"MAE": mae, "RMSE": rmse, "R2": r2})

        results.append({
            "Model": model_name,
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "R2": round(r2, 4),
            "Best_Params": grid.best_params_
        })

        print("Best parameters:", grid.best_params_)
        print(f"{model_name} Performance: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.4f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = grid.best_estimator_
            best_model_name = model_name

# ----------------------------
# Create model comparison output
# ----------------------------
results_df = pd.DataFrame(results).sort_values(by="RMSE", ascending=True)
print("\nModel Comparison Results:")
print(results_df)
print(f"\nSelected best model: {best_model_name} based on lowest RMSE")

# ----------------------------
# Save and register best model
# ----------------------------
os.makedirs("model_artifacts", exist_ok=True)
model_file = "model_artifacts/best_superkart_sales_model.joblib"
metrics_file = "model_artifacts/model_metrics.csv"

joblib.dump(best_model, model_file)
results_df.to_csv(metrics_file, index=False)

try:
    api.repo_info(repo_id=MODEL_REPO_ID, repo_type="model")
    print(f"Model repo '{MODEL_REPO_ID}' already exists. Using it.")
except RepositoryNotFoundError:
    create_repo(repo_id=MODEL_REPO_ID, repo_type="model", private=False, token=hf_token)

api.upload_file(
    path_or_fileobj=model_file,
    path_in_repo="best_superkart_sales_model.joblib",
    repo_id=MODEL_REPO_ID,
    repo_type="model",
    token=hf_token
)

api.upload_file(
    path_or_fileobj=metrics_file,
    path_in_repo="model_metrics.csv",
    repo_id=MODEL_REPO_ID,
    repo_type="model",
    token=hf_token
)

print("Best model:", best_model_name)
print("Best RMSE:", round(best_rmse, 2))
print("Best model and model comparison metrics registered successfully on Hugging Face Model Hub.")
