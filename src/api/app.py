import os
import sys
import subprocess
from pathlib import Path

import mlflow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.api.predict import load_production_model, predict
from src.mlflow_config import MLFLOW_TRACKING_URI

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

SUPPORTED_MODELS = {
    "xgboost": "xgboost-ticket-classifier",
}


# ============================================================
# REQUEST / RESPONSE MODELS
# ============================================================

class TicketRequest(BaseModel):
    """
    The fields available when a new ticket is created.
    Only the fields the model needs for prediction.
    """
    # Text fields (required)
    subject:     str
    description: str
    error_logs:  str = ""
    stack_trace: str = ""

    # Product info
    product:         str = ""
    product_version: str = ""
    product_module:  str = ""

    # Ticket metadata
    customer_tier:      str = "starter"
    priority:           str = "medium"
    severity:           str = "P3"
    channel:            str = "portal"
    customer_sentiment: str = "neutral"
    business_impact:    str = "low"
    environment:        str = "production"
    language:           str = "en"
    region:             str = "NA"

    # Numeric fields
    previous_tickets:           int = 0
    account_age_days:           int = 0
    account_monthly_value:      int = 0
    similar_issues_last_30_days: int = 0
    product_version_age_days:   int = 0
    affected_users:             int = 1
    attachments_count:          int = 0
    ticket_text_length:         int = 0

    # Binary fields
    contains_error_code:  bool = False
    contains_stack_trace: bool = False
    known_issue:          bool = False
    weekend_ticket:       bool = False
    after_hours:          bool = False

    # Tags
    tags: list[str] = []


class PredictionResponse(BaseModel):
    category:    str
    subcategory: str | None = None


class TrainRequest(BaseModel):
    model: str


class TrainResponse(BaseModel):
    model_name: str
    run_id:     str
    version:    str


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Ticket Classifier API",
    description="Predicts category and subcategory for support tickets "
                "using the production XGBoost model from MLflow.",
    version="1.0.0",
)


@app.on_event("startup")
def startup():
    """Load the production model when the API starts."""
    try:
        load_production_model()
    except Exception as e:
        print(f"ERROR: Could not load production model: {e}")
        print("Make sure you've run: python -m src.xgboost.register_production")
        raise


@app.post("/predict", response_model=PredictionResponse)
def predict_ticket(ticket: TicketRequest):
    """
    Predict the category and subcategory for a support ticket.

    Send a JSON body with the ticket fields. Only `subject` and
    `description` are required — all other fields have sensible defaults.
    """
    try:
        result = predict(ticket.model_dump())
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train", response_model=TrainResponse)
def train(request: TrainRequest):
    """
    Trigger a training run for the specified model.

    Supported models: "xgboost", "catboost".
    Runs training synchronously — the request will block until training completes.
    After training, promotes no alias automatically; call register_production separately.
    """
    if request.model not in SUPPORTED_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{request.model}'. Supported: {list(SUPPORTED_MODELS.keys())}",
        )

    train_script = PROJECT_ROOT / "src" / request.model / "train.py"
    result = subprocess.run(
        [sys.executable, str(train_script)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )

    if result.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail=f"Training failed:\n{result.stderr}",
        )

    # Retrieve the run_id and version of the just-completed run from MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    model_name = SUPPORTED_MODELS[request.model]
    experiment  = client.get_experiment_by_name(f"{request.model}-ticket-classifier")
    runs = client.search_runs(
        [experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    run_id = runs[0].info.run_id

    versions = client.search_model_versions(f"name='{model_name}'")
    latest_version = str(max(int(v.version) for v in versions))

    return TrainResponse(model_name=model_name, run_id=run_id, version=latest_version)
