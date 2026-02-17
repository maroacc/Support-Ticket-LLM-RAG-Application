"""
Generic MLflow logging and model registration utility.

Any model (XGBoost, CatBoost, etc.) can call log_and_register() to:
  - Log parameters, metrics, and artifacts to an MLflow run
  - Register a model version in the MLflow registry
"""
import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path

from mlflow_config import MLFLOW_TRACKING_URI, MLFLOW_ARTIFACT_ROOT


def log_and_register(
    experiment_name: str,
    model_name: str,
    run_name: str,
    params: dict,
    metrics: dict,
    artifact_dir: Path,
    plots_dir: Path | None = None,
):
    """
    Log params/metrics/artifacts to MLflow and register a model version.

    Args:
        experiment_name: MLflow experiment name (e.g. "catboost-ticket-classifier")
        model_name:      Registered model name in the registry
        run_name:        Display name for this run
        params:          Dict of parameters to log
        metrics:         Dict of metrics to log (values must be numeric)
        artifact_dir:    Path to directory containing model files (joblib, etc.)
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)

    print(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"MLflow experiment:   {experiment_name}")

    with mlflow.start_run(run_name=run_name) as run:

        # Log all parameters
        for key, value in params.items():
            mlflow.log_param(key, value)

        # Log all metrics
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        # Log the artifact directory
        mlflow.log_artifacts(str(artifact_dir), artifact_path="model")

        # Log plots as a separate top-level artifact folder
        if plots_dir is not None and plots_dir.exists():
            mlflow.log_artifacts(str(plots_dir), artifact_path="plots")

        run_id = run.info.run_id

    # Register model version
    client = MlflowClient()
    try:
        client.get_registered_model(model_name)
    except mlflow.exceptions.MlflowException:
        client.create_registered_model(model_name)

    result = client.create_model_version(
        name=model_name,
        source=f"runs:/{run_id}/model",
        run_id=run_id,
    )

    print(f"\n MLflow run logged: {run_id}")
    print(f"   Experiment: {experiment_name}")
    print(f"   Registered model: {model_name} version {result.version}")

    return run_id, result.version
