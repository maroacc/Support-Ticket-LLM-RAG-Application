import sys
import joblib
import pandas as pd

from pathlib import Path
from xgboost import XGBClassifier

# add parent (src/) to path so we can import shared modules
sys.path.append(str(Path(__file__).resolve().parent.parent))

from preprocessing import (preprocess, TFIDF_MAX_FEATURES, TEXT_COLS,
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
                X_val: pd.DataFrame,   y_val: pd.Series,
                target_name: str = "") -> XGBClassifier:
    """
    Train a single XGBoost classifier.

    We pass the validation set as eval_set so XGBoost can
    monitor performance during training and stop early if
    the validation score stops improving (early stopping).
    This prevents overfitting without having to tune n_estimators manually.
    """

    print(f"\n{'='*60}")
    print(f"  Training model: {target_name}")
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

    print(f"\n {target_name} model trained — best iteration: {model.best_iteration}")
    return model


# ============================================================
# SAVE MODEL AND ENCODERS
# ============================================================

def save_artifacts(category_model: XGBClassifier, subcategory_model: XGBClassifier,
                   feature_encoders: dict, target_encoders: dict,
                   version_dir: Path):
    """Save everything needed for inference to a versioned directory."""

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
        "cat_n_estimators": cat_model.n_estimators,
        "cat_max_depth": cat_model.max_depth,
        "cat_learning_rate": cat_model.learning_rate,
        "cat_early_stopping_rounds": cat_model.early_stopping_rounds,
        "cat_best_iteration": cat_model.best_iteration,
        "cat_num_classes": int(y_cat.nunique()),
        "cat_num_features": X_train.shape[1],
        # Subcategory model
        "sub_n_estimators": sub_model.n_estimators,
        "sub_max_depth": sub_model.max_depth,
        "sub_learning_rate": sub_model.learning_rate,
        "sub_early_stopping_rounds": sub_model.early_stopping_rounds,
        "sub_best_iteration": sub_model.best_iteration,
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
# MAIN — ties all steps together
# ============================================================

if __name__ == "__main__":

    # 1 — load raw data
    df = load_data(DATA_PATH)

    # 2 — preprocess (returns feature matrix X and encoded targets y)
    X, y, feature_encoders, target_encoders = preprocess(df, use_tfidf=USE_TFIDF)

    # ================================================================
    # MODEL 1 — predict category (5 classes)
    # ================================================================

    y_cat = y["category"]

    # 3 — split into train / validation / test
    X_train, X_val, X_test, y_cat_train, y_cat_val, y_cat_test = split_data(X, y_cat)

    # 4 — train category model
    cat_model = train_model(X_train, y_cat_train, X_val, y_cat_val,
                            target_name="category")

    # 5 — evaluate category model on test and train sets
    cat_test_metrics = evaluate_model(cat_model, X_test, y_cat_test,
                                      target_encoders, target_col="category", split="test")
    cat_train_metrics = evaluate_model(cat_model, X_train, y_cat_train,
                                       target_encoders, target_col="category", split="train")

    # ================================================================
    # MODEL 2 — predict subcategory (25 classes), using predicted
    #           category as an extra feature
    # ================================================================

    # 6 — get subcategory targets aligned with the same split indices
    y_sub_train = y["subcategory"].loc[X_train.index]
    y_sub_val   = y["subcategory"].loc[X_val.index]
    y_sub_test  = y["subcategory"].loc[X_test.index]

    # 7 — add category probabilities as features for each split
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

    # 8 — train subcategory model
    sub_model = train_model(X_train_sub, y_sub_train, X_val_sub, y_sub_val,
                            target_name="subcategory")

    # 9 — evaluate subcategory model on test and train sets
    sub_test_metrics = evaluate_model(sub_model, X_test_sub, y_sub_test,
                                      target_encoders, target_col="subcategory", split="test")
    sub_train_metrics = evaluate_model(sub_model, X_train_sub, y_sub_train,
                                       target_encoders, target_col="subcategory", split="train")

    # ================================================================
    # CONFUSION MATRICES
    # ================================================================

    plots_dir = MODELS_DIR / "latest_plots"
    subcat_to_cat = dict(zip(df["subcategory"], df["category"]))
    for target_col, model, X_split, y_split, split in [
        ("category",    cat_model, X_test,     y_cat_test,  "test"),
        ("category",    cat_model, X_train,    y_cat_train, "train"),
        ("subcategory", sub_model, X_test_sub, y_sub_test,  "test"),
        ("subcategory", sub_model, X_train_sub, y_sub_train, "train"),
    ]:
        y_pred = model.predict(X_split)
        plot_confusion_matrix(
            y_split, y_pred, target_encoders,
            target_col=target_col, split=split, save_dir=plots_dir,
            subcat_to_cat=subcat_to_cat if target_col == "subcategory" else None,
        )

    # ================================================================
    # FEATURE IMPORTANCE — diagnose what each model is using
    # ================================================================

    def print_feature_importance(model, feature_names, title, top_n=30):
        importance = model.feature_importances_
        feat_imp = sorted(zip(feature_names, importance),
                          key=lambda x: x[1], reverse=True)

        print(f"\n{'='*60}")
        print(f"  {title} — Top {top_n} features by importance")
        print(f"{'='*60}")
        for name, score in feat_imp[:top_n]:
            bar = "#" * int(score * 200)
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
    # SAVE EVERYTHING
    # ================================================================

    # 10 — save model artifacts to disk
    version_dir = MODELS_DIR / "latest"
    save_artifacts(cat_model, sub_model, feature_encoders,
                   target_encoders, version_dir)

    # 11 — log everything to MLflow and register model
    params = build_params(cat_model, sub_model, X_train, X_val, X_test, X,
                          y_cat, y_sub_train, cat_prob_cols)
    metrics = build_metrics(cat_test_metrics, sub_test_metrics,
                            cat_train_metrics, sub_train_metrics)

    # Feature importance metrics
    for name, score in zip(X_train.columns, cat_model.feature_importances_):
        metrics[f"cat_importance_{name}"] = float(score)
    for name, score in zip(X_train_sub.columns, sub_model.feature_importances_):
        metrics[f"sub_importance_{name}"] = float(score)

    run_id, version = log_and_register(
        experiment_name=EXPERIMENT_NAME,
        model_name=MODEL_NAME,
        run_name="xgboost_hierarchical",
        params=params,
        metrics=metrics,
        artifact_dir=version_dir,
        plots_dir=plots_dir,
    )

    print(f"   To promote to production, run:")
    print(f"   python -m src.xgboost.register_production")
