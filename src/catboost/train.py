"""
CatBoost hierarchical training: category model -> subcategory model.

Same approach as xgboost/train.py but uses CatBoostClassifier with
native categorical feature support. Uses shared mlflow_utils for logging.

Usage:
    python -m src.catboost.train
"""
import sys
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from catboost import CatBoostClassifier, Pool

# add parent (src/) to path so we can import shared modules
sys.path.append(str(Path(__file__).resolve().parent.parent))

from preprocessing import (preprocess, TFIDF_MAX_FEATURES, TEXT_COLS,
                           TOP_N_TAGS, CAT_COLS, NUMERIC_COLS, BINARY_COLS)
from data_utils import load_data, split_data, DATA_PATH, RANDOM_SEED
from evaluate import evaluate_model
from mlflow_utils import log_and_register


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT    = Path(__file__).resolve().parent.parent.parent
MODELS_DIR      = PROJECT_ROOT / "models" / "catboost"
USE_TFIDF       = False
EXPERIMENT_NAME = "catboost-ticket-classifier"
MODEL_NAME      = "catboost-ticket-classifier"


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(X_train: pd.DataFrame, y_train: pd.Series,
                X_val: pd.DataFrame,   y_val: pd.Series,
                cat_feature_indices: list,
                target_name: str = "") -> CatBoostClassifier:
    """
    Train a single CatBoost classifier.

    CatBoost handles categorical features natively — we pass them
    via cat_features parameter. eval_set monitors validation loss
    for early stopping.
    """

    print(f"\n{'='*60}")
    print(f"  Training CatBoost model: {target_name}")
    print(f"  Features: {X_train.shape[1]} | Classes: {y_train.nunique()}")
    print(f"{'='*60}")

    model = CatBoostClassifier(
        loss_function="MultiClass",
        iterations=500,
        learning_rate=0.1,
        depth=6,
        early_stopping_rounds=20,
        random_seed=RANDOM_SEED,
        verbose=50,
    )

    train_pool = Pool(X_train, y_train, cat_features=cat_feature_indices)
    val_pool   = Pool(X_val,   y_val,   cat_features=cat_feature_indices)

    model.fit(train_pool, eval_set=val_pool)

    print(f"\n {target_name} model trained — best iteration: {model.best_iteration_}")
    return model


# ============================================================
# SAVE ARTIFACTS
# ============================================================

def save_artifacts(category_model, subcategory_model,
                   feature_encoders: dict, target_encoders: dict,
                   version_dir: Path):
    version_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(category_model,    version_dir / "category_model.joblib")
    joblib.dump(subcategory_model, version_dir / "subcategory_model.joblib")
    joblib.dump(feature_encoders,  version_dir / "feature_encoders.joblib")
    joblib.dump(target_encoders,   version_dir / "target_encoders.joblib")

    print(f"\n Artifacts saved to {version_dir}/")


# ============================================================
# BUILD MLFLOW PARAMS AND METRICS
# ============================================================

def build_params(cat_model, sub_model, X_train, X_val, X_test, X,
                 y_cat, y_sub_train, cat_prob_cols):
    return {
        # Data
        "data_source": DATA_PATH,
        "total_samples": len(X),
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "random_seed": RANDOM_SEED,
        # Preprocessing
        "use_tfidf": USE_TFIDF,
        "tfidf_max_features": TFIDF_MAX_FEATURES if USE_TFIDF else None,
        "top_n_tags": TOP_N_TAGS,
        # Category model
        "cat_iterations": cat_model.tree_count_,
        "cat_depth": cat_model.get_param("depth"),
        "cat_learning_rate": cat_model.get_param("learning_rate"),
        "cat_best_iteration": cat_model.best_iteration_,
        "cat_num_classes": int(y_cat.nunique()),
        "cat_num_features": X_train.shape[1],
        # Subcategory model
        "sub_iterations": sub_model.tree_count_,
        "sub_depth": sub_model.get_param("depth"),
        "sub_learning_rate": sub_model.get_param("learning_rate"),
        "sub_best_iteration": sub_model.best_iteration_,
        "sub_num_classes": int(y_sub_train.nunique()),
        "sub_num_features": X_train.shape[1] + len(cat_prob_cols),
    }


def build_metrics(cat_test_metrics, sub_test_metrics,
                  cat_train_metrics, sub_train_metrics):
    metrics = {}

    for prefix, report in [("cat_test", cat_test_metrics),
                           ("cat_train", cat_train_metrics),
                           ("sub_test", sub_test_metrics),
                           ("sub_train", sub_train_metrics)]:
        metrics[f"{prefix}_weighted_f1"] = report["weighted avg"]["f1-score"]
        metrics[f"{prefix}_weighted_precision"] = report["weighted avg"]["precision"]
        metrics[f"{prefix}_weighted_recall"] = report["weighted avg"]["recall"]
        metrics[f"{prefix}_accuracy"] = report["accuracy"]

        # Per-class F1
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
    cat_feature_indices = feature_encoders["cat_feature_indices"]

    # ================================================================
    # MODEL 1 — predict category
    # ================================================================

    y_cat = y["category"]
    X_train, X_val, X_test, y_cat_train, y_cat_val, y_cat_test = split_data(X, y_cat)

    cat_model = train_model(X_train, y_cat_train, X_val, y_cat_val,
                            cat_feature_indices, target_name="category")

    cat_test_metrics = evaluate_model(cat_model, X_test, y_cat_test,
                                      target_encoders, target_col="category", split="test")
    cat_train_metrics = evaluate_model(cat_model, X_train, y_cat_train,
                                       target_encoders, target_col="category", split="train")

    # ================================================================
    # MODEL 2 — predict subcategory (with category probs as features)
    # ================================================================

    y_sub_train = y["subcategory"].loc[X_train.index]
    y_sub_val   = y["subcategory"].loc[X_val.index]
    y_sub_test  = y["subcategory"].loc[X_test.index]

    cat_encoder = target_encoders["category"]
    cat_prob_cols = [f"cat_prob_{name}" for name in cat_encoder.classes_]

    def add_category_probs(X_split):
        X_out = X_split.copy()
        probs = cat_model.predict_proba(X_split)
        for i, col in enumerate(cat_prob_cols):
            X_out[col] = probs[:, i]
        return X_out

    X_train_sub = add_category_probs(X_train)
    X_val_sub   = add_category_probs(X_val)
    X_test_sub  = add_category_probs(X_test)

    # Subcategory features include the cat_prob_ columns which are numeric,
    # so categorical indices stay the same (they're at the front)
    sub_model = train_model(X_train_sub, y_sub_train, X_val_sub, y_sub_val,
                            cat_feature_indices, target_name="subcategory")

    sub_test_metrics = evaluate_model(sub_model, X_test_sub, y_sub_test,
                                      target_encoders, target_col="subcategory", split="test")
    sub_train_metrics = evaluate_model(sub_model, X_train_sub, y_sub_train,
                                       target_encoders, target_col="subcategory", split="train")

    # ================================================================
    # FEATURE IMPORTANCE
    # ================================================================

    def print_feature_importance(model, feature_names, title, top_n=30):
        importance = model.get_feature_importance()
        feat_imp = sorted(zip(feature_names, importance),
                          key=lambda x: x[1], reverse=True)

        print(f"\n{'='*60}")
        print(f"  {title} — Top {top_n} features by importance")
        print(f"{'='*60}")
        for name, score in feat_imp[:top_n]:
            bar = "#" * int(score * 2)
            print(f"  {score:.4f}  {bar}  {name}")

        cat_feats = [(n, s) for n, s in feat_imp if n.startswith("cat_prob_")]
        if cat_feats:
            print(f"\n  Category probability features:")
            for name, score in cat_feats:
                print(f"    {score:.4f}  {name}")

        zero_count = sum(1 for _, s in feat_imp if s == 0)
        print(f"\n  Features with zero importance: {zero_count}/{len(feat_imp)}")

    print_feature_importance(cat_model, list(X_train.columns),
                             "Category model")
    print_feature_importance(sub_model, list(X_train_sub.columns),
                             "Subcategory model")

    # ================================================================
    # SAVE AND LOG
    # ================================================================

    # Save artifacts to disk
    version_dir = MODELS_DIR / "latest"
    save_artifacts(cat_model, sub_model, feature_encoders,
                   target_encoders, version_dir)

    # Log to MLflow
    params = build_params(cat_model, sub_model, X_train, X_val, X_test, X,
                          y_cat, y_sub_train, cat_prob_cols)
    metrics = build_metrics(cat_test_metrics, sub_test_metrics,
                            cat_train_metrics, sub_train_metrics)

    # Feature importance metrics
    for name, score in zip(X_train.columns, cat_model.get_feature_importance()):
        metrics[f"cat_importance_{name}"] = float(score)
    for name, score in zip(X_train_sub.columns, sub_model.get_feature_importance()):
        metrics[f"sub_importance_{name}"] = float(score)

    run_id, version = log_and_register(
        experiment_name=EXPERIMENT_NAME,
        model_name=MODEL_NAME,
        run_name="catboost_hierarchical",
        params=params,
        metrics=metrics,
        artifact_dir=version_dir,
    )

    print(f"\n   To promote to production, run:")
    print(f"   python -m src.catboost.register_production")
