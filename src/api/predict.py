"""
Prediction module: loads the production model from MLflow and
preprocesses a single ticket for inference.

Supports both XGBoost and CatBoost models. The model type is
auto-detected from the saved feature_encoders (CatBoost models
have a "cat_feature_names" key).
"""
import joblib
import mlflow
import pandas as pd

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

from src.mlflow_config import MLFLOW_TRACKING_URI
from src.xgboost.preprocessing import (
    ONE_HOT_COLS, LABEL_ENCODE_COLS, NUMERIC_COLS, BINARY_COLS, TEXT_COLS
)


# ============================================================
# MODULE STATE — loaded once on startup
# ============================================================

_category_model    = None
_subcategory_model = None
_feature_encoders  = None
_target_encoders   = None
_is_loaded         = False
_model_type        = None  # "xgboost" or "catboost"


# ============================================================
# LOAD PRODUCTION MODEL
# ============================================================

def load_production_model(model_name: str = "xgboost-ticket-classifier"):
    """
    Load the production model from MLflow registry.

    Args:
        model_name: Registered model name. Use "xgboost-ticket-classifier"
                    or "catboost-ticket-classifier".
    """
    global _category_model, _subcategory_model
    global _feature_encoders, _target_encoders, _is_loaded, _model_type

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    # Get the model version tagged as "production"
    version_info = client.get_model_version_by_alias(model_name, "production")
    run_id = version_info.run_id
    version = version_info.version

    print(f"Loading production model: {model_name} version {version}")
    print(f"  MLflow run ID: {run_id}")

    # Download the artifacts to a local directory
    artifact_dir = Path(client.download_artifacts(run_id, "model"))

    # Load the individual joblib files
    _category_model    = joblib.load(artifact_dir / "category_model.joblib")
    _subcategory_model = joblib.load(artifact_dir / "subcategory_model.joblib")
    _feature_encoders  = joblib.load(artifact_dir / "feature_encoders.joblib")
    _target_encoders   = joblib.load(artifact_dir / "target_encoders.joblib")

    # Auto-detect model type
    _model_type = "catboost" if "cat_feature_names" in _feature_encoders else "xgboost"
    _is_loaded = True

    print(f"  Loaded: category_model, subcategory_model, encoders")
    print(f"  Detected model type: {_model_type}")


# ============================================================
# PREPROCESS — XGBOOST PATH
# ============================================================

def _preprocess_ticket_xgboost(ticket: dict) -> pd.DataFrame:
    """
    XGBoost preprocessing: label encode + one-hot → integer features.
    Uses model.get_booster().feature_names for column alignment.
    """
    expected_cols = _category_model.get_booster().feature_names
    tags = ticket.get("tags", []) or []

    row = {col: 0 for col in expected_cols}

    # One-hot columns
    for col in ONE_HOT_COLS:
        val = str(ticket.get(col, "unknown"))
        col_name = f"{col}_{val}"
        if col_name in row:
            row[col_name] = 1

    # Label encoding (using saved encoders)
    for col in LABEL_ENCODE_COLS:
        encoder = _feature_encoders.get(col)
        if encoder is None or col not in row:
            continue
        val = str(ticket.get(col, "unknown"))
        if val in encoder.classes_:
            row[col] = int(encoder.transform([val])[0])
        else:
            row[col] = 0

    # Numeric columns
    for col in NUMERIC_COLS:
        if col in row:
            row[col] = ticket.get(col, 0)

    # Binary columns
    for col in BINARY_COLS:
        if col in row:
            row[col] = int(ticket.get(col, False))

    # Tags multi-hot encoding
    for col_name in expected_cols:
        if col_name.startswith("tag_"):
            tag_name = col_name.replace("tag_", "", 1)
            row[col_name] = int(tag_name in tags)

    # TF-IDF
    tfidf_vectorizer: TfidfVectorizer = _feature_encoders.get("tfidf_vectorizer")
    if tfidf_vectorizer is not None:
        text = " ".join(str(ticket.get(col, "") or "") for col in TEXT_COLS)
        tfidf_matrix = tfidf_vectorizer.transform([text])
        feature_names = tfidf_vectorizer.get_feature_names_out()
        tfidf_values = tfidf_matrix.toarray()[0]
        for word, value in zip(feature_names, tfidf_values):
            col_name = f"tfidf_{word}"
            if col_name in row:
                row[col_name] = value

    df = pd.DataFrame([row], columns=expected_cols)
    return df


# ============================================================
# PREPROCESS — CATBOOST PATH
# ============================================================

def _preprocess_ticket_catboost(ticket: dict) -> pd.DataFrame:
    """
    CatBoost preprocessing: categorical values as strings (no encoding),
    numeric/binary/tfidf/tags as numbers.
    Uses model.feature_names_ for column alignment.
    """
    expected_cols = _category_model.feature_names_
    cat_feature_names = _feature_encoders.get("cat_feature_names", [])
    tags = ticket.get("tags", []) or []

    row = {col: 0 for col in expected_cols}

    # Categorical columns — pass as strings
    for col in cat_feature_names:
        if col in row:
            row[col] = str(ticket.get(col, "unknown"))

    # Numeric columns
    for col in NUMERIC_COLS:
        if col in row:
            row[col] = ticket.get(col, 0)

    # Binary columns
    for col in BINARY_COLS:
        if col in row:
            row[col] = int(ticket.get(col, False))

    # Tags multi-hot encoding
    for col_name in expected_cols:
        if col_name.startswith("tag_"):
            tag_name = col_name.replace("tag_", "", 1)
            row[col_name] = int(tag_name in tags)

    # TF-IDF
    tfidf_vectorizer: TfidfVectorizer = _feature_encoders.get("tfidf_vectorizer")
    if tfidf_vectorizer is not None:
        text = " ".join(str(ticket.get(col, "") or "") for col in TEXT_COLS)
        tfidf_matrix = tfidf_vectorizer.transform([text])
        feature_names = tfidf_vectorizer.get_feature_names_out()
        tfidf_values = tfidf_matrix.toarray()[0]
        for word, value in zip(feature_names, tfidf_values):
            col_name = f"tfidf_{word}"
            if col_name in row:
                row[col_name] = value

    df = pd.DataFrame([row], columns=expected_cols)
    return df


# ============================================================
# PREDICT
# ============================================================

def predict(ticket: dict) -> dict:
    """
    Predict category and subcategory for a single ticket.

    Args:
        ticket: dict with the ticket fields available at creation time.

    Returns:
        {"category": "Technical Issue", "subcategory": "Configuration"}
    """
    if not _is_loaded:
        raise RuntimeError("Call load_production_model() first")

    # Preprocess based on model type
    if _model_type == "catboost":
        X = _preprocess_ticket_catboost(ticket)
    else:
        X = _preprocess_ticket_xgboost(ticket)

    # Predict category
    cat_pred = _category_model.predict(X)[0]
    cat_label = _target_encoders["category"].inverse_transform([int(cat_pred)])[0]

    # Add category probabilities as features for subcategory model
    cat_probs = _category_model.predict_proba(X)
    cat_encoder = _target_encoders["category"]
    cat_prob_cols = [f"cat_prob_{name}" for name in cat_encoder.classes_]

    X_sub = X.copy()
    for i, col in enumerate(cat_prob_cols):
        X_sub[col] = cat_probs[0][i]

    # Predict subcategory
    sub_pred = _subcategory_model.predict(X_sub)[0]
    sub_label = _target_encoders["subcategory"].inverse_transform([int(sub_pred)])[0]

    return {
        "category": cat_label,
        "subcategory": sub_label,
    }
