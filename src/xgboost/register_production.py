"""
One-time script to register the existing trained model in MLflow
and set it as the production model.

Usage:
    python -m src.xgboost.register_production

What it does:
    1. Creates an MLflow run with the existing model artifacts (20260213_182053)
    2. Registers it in the model registry using the client API
    3. Sets the alias "production" on that version

After this, the API loads the production model by looking up the
run tagged with the "production" alias and downloading its artifacts.
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# The best existing model directory
MODEL_DIR       = PROJECT_ROOT / "models" / "xgboost" / "20260213_182053"
EXPERIMENT_NAME = "xgboost-ticket-classifier"
MODEL_NAME      = "xgboost-ticket-classifier"


# ============================================================
# MAIN
# ============================================================

def main():
    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient()

    print(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"Registering model from: {MODEL_DIR}")

    # Create a run and log the existing artifacts
    with mlflow.start_run(run_name="initial_production_model") as run:

        # Log some basic params so the run is identifiable
        mlflow.log_param("source", "pre-existing model")
        mlflow.log_param("original_version", "20260213_182053")
        mlflow.log_param("model_type", "xgboost_hierarchical")

        # Log the model files as artifacts
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

    # Create a model version pointing to this run's artifacts
    version = client.create_model_version(
        name=MODEL_NAME,
        source=f"runs:/{run_id}/model",
        run_id=run_id,
    )
    print(f" Created model version: {version.version}")

    # Set the "production" alias on this version
    client.set_registered_model_alias(
        MODEL_NAME, "production", version.version
    )
    print(f" Set alias 'production' -> version {version.version}")

    print(f"\n Done! The API will load artifacts from run: {run_id}")


if __name__ == "__main__":
    main()
