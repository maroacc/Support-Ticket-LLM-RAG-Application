import os
import sys
import subprocess
from pathlib import Path

import mlflow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.api.predict import load_production_model, predict
from src.mlflow_config import MLFLOW_TRACKING_URI
from src.rag import solution_finder

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

MODEL_NAME = "xgboost-ticket-classifier"
TRAIN_SCRIPT = "src/xgboost/train.py"
EXPERIMENT_NAME = "xgboost-ticket-classifier"


# ============================================================
# REQUEST / RESPONSE MODELS
# ============================================================

class TicketRequest(BaseModel):
    """
    The fields available when a new ticket is created.
    Only the fields the model needs for prediction.
    """

    model_config = {
        "json_schema_extra": {
            "example": {
                "subject": "Request: Add bulk operation support to CloudBackup Enterprise",
                "description": "We would like to request a feature for CloudBackup Enterprise that allows bulk operations. Currently, we have to process items one by one, which is time-consuming. Having bulk support would greatly improve our workflow efficiency.",
                "error_logs": "",
                "stack_trace": "",
                "product": "CloudBackup Enterprise",
                "product_version": "4.5.10",
                "product_module": "encryption_layer",
                "customer_tier": "starter",
                "priority": "critical",
                "severity": "P2",
                "channel": "portal",
                "customer_sentiment": "frustrated",
                "business_impact": "high",
                "environment": "production",
                "language": "de",
                "region": "APAC",
                "previous_tickets": 9,
                "account_age_days": 696,
                "account_monthly_value": 127,
                "similar_issues_last_30_days": 130,
                "product_version_age_days": 24,
                "affected_users": 222,
                "attachments_count": 4,
                "ticket_text_length": 230,
                "contains_error_code": False,
                "contains_stack_trace": False,
                "known_issue": False,
                "weekend_ticket": False,
                "after_hours": False,
                "tags": ["error", "api", "integration", "timeout", "bug"]
            }
        }
    }

    # Text fields
    subject:     str
    description: str
    error_logs:  str
    stack_trace: str

    # Product info
    product:         str
    product_version: str
    product_module:  str

    # Ticket metadata
    customer_tier:      str
    priority:           str
    severity:           str
    channel:            str
    customer_sentiment: str
    business_impact:    str
    environment:        str
    language:           str
    region:             str

    # Numeric fields
    previous_tickets:            int
    account_age_days:            int
    account_monthly_value:       int
    similar_issues_last_30_days: int
    product_version_age_days:    int
    affected_users:              int
    attachments_count:           int
    ticket_text_length:          int

    # Binary fields
    contains_error_code:  bool
    contains_stack_trace: bool
    known_issue:          bool
    weekend_ticket:       bool
    after_hours:          bool

    # Tags
    tags: list[str]


class RAGRequest(BaseModel):
    """
    Fields needed by the RAG pipeline.
    Call /predict first to get the category, then include it here.
    """

    model_config = {
        "json_schema_extra": {
            "example": {
                "subject": "Request: Add bulk operation support to CloudBackup Enterprise",
                "description": "We would like to request a feature for CloudBackup Enterprise that allows bulk operations. Currently, we have to process items one by one, which is time-consuming. Having bulk support would greatly improve our workflow efficiency.",
                "error_logs": "",
                "product": "CloudBackup Enterprise",
                "product_version": "4.5.10",
                "product_module": "encryption_layer",
                "category": "Feature Request",
            }
        }
    }

    subject:         str
    description:     str
    error_logs:      str
    product:         str
    product_version: str
    product_module:  str
    category:        str


class PredictionResponse(BaseModel):
    category: str


class TrainResponse(BaseModel):
    model_name: str
    run_id:     str
    version:    str


class KBArticle(BaseModel):
    article:      str
    success_rate: float | None = None


class Solution(BaseModel):
    resolution:          str
    resolution_code:     str | None = None
    resolution_template: str | None = None
    resolution_helpful:  bool | None = None
    kb_articles:         list[KBArticle] = []


class RAGResult(BaseModel):
    ticket_id:        str
    subject:          str
    similarity_score: float
    match_ratio:      float
    final_score:      float
    solution:         Solution


class RAGResponse(BaseModel):
    category: str
    results:  list[RAGResult]


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
    """Load the production model and RAG artifacts when the API starts."""
    try:
        load_production_model()
    except Exception as e:
        print(f"ERROR: Could not load production model: {e}")
        print("Make sure you've run: python -m src.xgboost.register_production")
        raise

    try:
        solution_finder.load()
    except Exception as e:
        print(f"WARNING: Could not load RAG artifacts: {e}")
        print("The /rag endpoint will be unavailable.")
        print("Run the build scripts in src/rag/ to generate the required files.")


@app.post("/predict", response_model=PredictionResponse)
def predict_ticket(ticket: TicketRequest):
    """
    Predict the category and subcategory for a support ticket.
    Send a JSON body with the ticket fields.
    """
    try:
        result = predict(ticket.model_dump())
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag", response_model=RAGResponse)
def rag(ticket: RAGRequest, top_k: int = 5):
    """
    Find similar historical tickets and return their solutions.

    Expects the category predicted by /predict to be included in the request body.
    The category is passed to the retrieval system as an additional matching signal.

    `top_k` (query parameter, default 5) controls how many results to return.
    """
    if solution_finder._knowledge_graph is None:
        raise HTTPException(
            status_code=503,
            detail="RAG artifacts not loaded. Run the build scripts in src/rag/ first.",
        )

    ticket_dict = ticket.model_dump()

    try:
        raw_results = solution_finder.find_solutions(ticket_dict, top_k=top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    results = [
        RAGResult(
            ticket_id        = r["ticket_id"],
            subject          = r["ticket_data"].get("subject", ""),
            similarity_score = r["similarity_score"],
            match_ratio      = r["match_ratio"],
            final_score      = r["final_score"],
            solution         = Solution(
                resolution          = r["solution"]["resolution"] or "",
                resolution_code     = r["solution"]["resolution_code"],
                resolution_template = r["solution"]["resolution_template"],
                resolution_helpful  = r["solution"]["resolution_helpful"],
                kb_articles         = [
                    KBArticle(**kb) for kb in r["solution"]["kb_articles"]
                ],
            ),
        )
        for r in raw_results
    ]

    return RAGResponse(category=ticket.category, results=results)


@app.post("/train", response_model=TrainResponse)
def train():
    """
    Trigger an XGBoost training run.

    Runs training synchronously  the request will block until training completes.
    After training, the new version is not promoted automatically;
    call register_production separately.
    """
    result = subprocess.run(
        [sys.executable, TRAIN_SCRIPT],
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

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    runs = client.search_runs(
        [experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    run_id = runs[0].info.run_id

    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    latest_version = str(max(int(v.version) for v in versions))

    return TrainResponse(model_name=MODEL_NAME, run_id=run_id, version=latest_version)
