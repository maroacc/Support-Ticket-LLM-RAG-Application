"""
Register the CatBoost model as the production model in MLflow.

Usage:
    python -m src.catboost.register_production

What it does:
    1. Creates an MLflow run with the existing model artifacts
    2. Registers it in the model registry
    3. Sets the alias "production" on that version
"""
import sys
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

# add parent (src/) to path so we can import shared modules
sys.path.append(str(Path(__file__).resolve().parent.parent))

from mlflow_config import MLFLOW_TRACKING_URI


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT    = Path(__file__).resolve().parent.parent.parent
MODEL_DIR       = PROJECT_ROOT / "models" / "catboost" / "latest"
EXPERIMENT_NAME = "catboost-ticket-classifier"
MODEL_NAME      = "catboost-ticket-classifier"


# ============================================================
# MAIN
# ============================================================

def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient()

    print(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"Registering model from: {MODEL_DIR}")

    # Create a run and log the existing artifacts
    with mlflow.start_run(run_name="register_production_catboost") as run:
        mlflow.log_param("source", "pre-existing model")
        mlflow.log_param("model_type", "catboost_hierarchical")
        mlflow.log_artifacts(str(MODEL_DIR), artifact_path="model")
        run_id = run.info.run_id
        print(f"\n Run created: {run_id}")

    # Ensure the registered model exists
    try:
        client.get_registered_model(MODEL_NAME)
        print(f" Registered model '{MODEL_NAME}' already exists")
    except mlflow.exceptions.MlflowException:
        client.create_registered_model(MODEL_NAME)
        print(f" Created registered model: {MODEL_NAME}")

    # Create a model version
    version = client.create_model_version(
        name=MODEL_NAME,
        source=f"runs:/{run_id}/model",
        run_id=run_id,
    )
    print(f" Created model version: {version.version}")

    # Set the "production" alias
    client.set_registered_model_alias(
        MODEL_NAME, "production", version.version
    )
    print(f" Set alias 'production' -> version {version.version}")
    print(f"\n Done! The API will load artifacts from run: {run_id}")


if __name__ == "__main__":
    main()
