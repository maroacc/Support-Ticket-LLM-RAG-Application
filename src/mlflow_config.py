from pathlib import Path


# ============================================================
# MLFLOW CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# SQLite backend  supports model registry, works on Windows and Linux
# Resolves to something like: sqlite:///C:/Users/.../mlruns.db
MLFLOW_DB_PATH      = PROJECT_ROOT / "mlruns.db"
MLFLOW_TRACKING_URI = f"sqlite:///{str(MLFLOW_DB_PATH).replace(chr(92), '/')}"

# Where MLflow stores artifacts (model files, encoders, etc.)
MLFLOW_ARTIFACT_ROOT = str(PROJECT_ROOT / "mlartifacts").replace("\\", "/")

# Per-model experiment and model names are defined in each model's train.py.
# This file only contains shared infrastructure config.
