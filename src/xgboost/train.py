import sys
import joblib
import pandas as pd

from pathlib import Path
from xgboost import XGBClassifier

# add parent (src/) to path so we can import shared modules
sys.path.append(str(Path(__file__).resolve().parent.parent))

from preprocessing import (preprocess, TFIDF_MAX_FEATURES,
                           TOP_N_TAGS, ONE_HOT_COLS, LABEL_ENCODE_COLS,
                           NUMERIC_COLS, BINARY_COLS)
from data_utils import load_data, split_data, DATA_PATH, RANDOM_SEED
from evaluate import evaluate_model, plot_confusion_matrix
from mlflow_utils import log_and_register


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT    = Path(__file__).resolve().parent.parent.parent
MODELS_DIR      = PROJECT_ROOT / "models" / "xgboost"
USE_TFIDF       = False
EXPERIMENT_NAME = "xgboost-ticket-classifier"
MODEL_NAME      = "xgboost-ticket-classifier"


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(X_train: pd.DataFrame, y_train: pd.Series,
                X_val: pd.DataFrame,   y_val: pd.Series) -> XGBClassifier:
    print(f"\n{'='*60}")
    print(f"  Training category model")
    print(f"  Features: {X_train.shape[1]} | Classes: {y_train.nunique()}")
    print(f"{'='*60}")

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=y_train.nunique(),
        n_estimators=500,
        learning_rate=0.1,
        max_depth=6,
        early_stopping_rounds=20,
        eval_metric="mlogloss",
        random_state=RANDOM_SEED,
        verbosity=1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=50
    )

    print(f"\n Category model trained — best iteration: {model.best_iteration}")
    return model


# ============================================================
# SAVE ARTIFACTS
# ============================================================

def save_artifacts(model: XGBClassifier, feature_encoders: dict,
                   target_encoders: dict, version_dir: Path):
    version_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model,            version_dir / "category_model.joblib")
    joblib.dump(feature_encoders, version_dir / "feature_encoders.joblib")
    joblib.dump(target_encoders,  version_dir / "target_encoders.joblib")

    print(f"\n Artifacts saved to {version_dir}/")


# ============================================================
# BUILD MLFLOW PARAMS AND METRICS
# ============================================================

def build_params(model, X_train, X_val, X_test, X, y_cat):
    return {
        "data_source":             DATA_PATH,
        "total_samples":           len(X),
        "train_samples":           len(X_train),
        "val_samples":             len(X_val),
        "test_samples":            len(X_test),
        "random_seed":             RANDOM_SEED,
        "use_tfidf":               USE_TFIDF,
        "tfidf_max_features":      TFIDF_MAX_FEATURES if USE_TFIDF else None,
        "top_n_tags":              TOP_N_TAGS,
        "n_estimators":            model.n_estimators,
        "max_depth":               model.max_depth,
        "learning_rate":           model.learning_rate,
        "early_stopping_rounds":   model.early_stopping_rounds,
        "best_iteration":          model.best_iteration,
        "num_classes":             int(y_cat.nunique()),
        "num_features":            X_train.shape[1],
    }


def build_metrics(test_metrics, train_metrics):
    metrics = {}

    for prefix, report in [("test", test_metrics), ("train", train_metrics)]:
        metrics[f"{prefix}_weighted_f1"]        = report["weighted avg"]["f1-score"]
        metrics[f"{prefix}_weighted_precision"] = report["weighted avg"]["precision"]
        metrics[f"{prefix}_weighted_recall"]    = report["weighted avg"]["recall"]
        metrics[f"{prefix}_accuracy"]           = report["accuracy"]

        for class_name, class_data in report.items():
            if class_name in ("accuracy", "macro avg", "weighted avg"):
                continue
            metrics[f"{prefix}_f1_{class_name}"] = class_data["f1-score"]

    return metrics


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # 1 — load raw data
    df = load_data(DATA_PATH)

    # 2 — preprocess
    X, y, feature_encoders, target_encoders = preprocess(df, use_tfidf=USE_TFIDF)
    feature_encoders["has_subcategory_model"] = False

    # 3 — split
    y_cat = y["category"]
    X_train, X_val, X_test, y_cat_train, y_cat_val, y_cat_test = split_data(X, y_cat)

    # 4 — train
    model = train_model(X_train, y_cat_train, X_val, y_cat_val)

    # 5 — evaluate
    test_metrics  = evaluate_model(model, X_test,  y_cat_test,
                                   target_encoders, target_col="category", split="test")
    train_metrics = evaluate_model(model, X_train, y_cat_train,
                                   target_encoders, target_col="category", split="train")

    # 6 — confusion matrices
    plots_dir = MODELS_DIR / "latest_plots"
    for X_split, y_split, split in [(X_test, y_cat_test, "test"),
                                     (X_train, y_cat_train, "train")]:
        y_pred = model.predict(X_split)
        plot_confusion_matrix(y_split, y_pred, target_encoders,
                              target_col="category", split=split, save_dir=plots_dir)

    # 7 — feature importance
    importance = model.feature_importances_
    feat_imp   = sorted(zip(X_train.columns, importance),
                        key=lambda x: x[1], reverse=True)
    print(f"\n{'='*60}")
    print(f"  Category model — Top 30 features by importance")
    print(f"{'='*60}")
    for name, score in feat_imp[:30]:
        print(f"  {score:.4f}  {'#' * int(score * 200)}  {name}")
    zero_count = sum(1 for _, s in feat_imp if s == 0)
    print(f"\n  Features with zero importance: {zero_count}/{len(feat_imp)}")

    # 8 — save artifacts
    version_dir = MODELS_DIR / "latest"
    save_artifacts(model, feature_encoders, target_encoders, version_dir)

    # 9 — log to MLflow and register
    params  = build_params(model, X_train, X_val, X_test, X, y_cat)
    metrics = build_metrics(test_metrics, train_metrics)

    for name, score in zip(X_train.columns, model.feature_importances_):
        metrics[f"importance_{name}"] = float(score)

    run_id, version = log_and_register(
        experiment_name=EXPERIMENT_NAME,
        model_name=MODEL_NAME,
        run_name="xgboost_category",
        params=params,
        metrics=metrics,
        artifact_dir=version_dir,
        plots_dir=plots_dir,
    )

    print(f"   To promote to production, run:")
    print(f"   python -m src.xgboost.register_production")
