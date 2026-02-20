import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_predictions(y_true, y_pred, target_encoders: dict,
                         target_col: str = "category",
                         split: str = "test") -> dict:
    """
    Evaluate predictions on a given data split.

    Accepts raw integer arrays (works with any model type).

    - Precision : of all tickets predicted as class X, how many were actually X?
    - Recall    : of all tickets that are class X, how many did we catch?
    - F1        : harmonic mean of precision and recall (the headline metric)

    Args:
        split: Which data split is being evaluated ("train", "val", or "test").
    """

    encoder = target_encoders[target_col]
    y_true_labels = encoder.inverse_transform(np.asarray(y_true).astype(int))
    y_pred_labels = encoder.inverse_transform(np.asarray(y_pred).astype(int))

    report = classification_report(y_true_labels, y_pred_labels, output_dict=True)

    print(f"\n CLASSIFICATION REPORT  {target_col} ({split} set)\n")
    print(classification_report(y_true_labels, y_pred_labels))

    return report


def plot_confusion_matrix(y_true, y_pred, target_encoders: dict,
                          target_col: str = "category",
                          split: str = "test",
                          save_dir: Path | None = None,
                          subcat_to_cat: dict | None = None):
    """
    Plot and optionally save a confusion matrix heatmap.

    Args:
        y_true:           Ground-truth integer labels.
        y_pred:           Predicted integer labels.
        target_encoders:  Dict of fitted LabelEncoders (keyed by target_col).
        target_col:       Which target ("category" or "subcategory").
        split:            Data split name (used in the title and filename).
        save_dir:         If provided, saves the figure as a PNG in this directory.
        subcat_to_cat:    Optional dict mapping subcategory -> category. When provided,
                          labels are prefixed with their category and sorted so that
                          subcategories of the same category appear next to each other.
    """
    encoder = target_encoders[target_col]
    y_true_labels = encoder.inverse_transform(np.asarray(y_true).astype(int))
    y_pred_labels = encoder.inverse_transform(np.asarray(y_pred).astype(int))

    labels = list(encoder.classes_)

    # Group subcategories by category when mapping is provided
    if subcat_to_cat is not None:
        label_map = {sub: f"{cat}  {sub}" for sub, cat in subcat_to_cat.items()}
        labels = sorted(labels, key=lambda s: (subcat_to_cat.get(s, ""), s))
        display_labels = [label_map.get(l, l) for l in labels]
    else:
        display_labels = labels

    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=labels)

    # Scale figure size based on number of classes
    size = max(8, len(labels) * 0.6)
    fig, ax = plt.subplots(figsize=(size, size))

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=display_labels, yticklabels=display_labels, ax=ax,
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix  {target_col} ({split} set)")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        filepath = save_dir / f"confusion_matrix_{target_col}_{split}.png"
        fig.savefig(filepath, dpi=150)
        print(f"  Confusion matrix saved to {filepath}")

    plt.close(fig)


def evaluate_model(model, X, y, target_encoders: dict,
                   target_col: str = "category",
                   split: str = "test") -> dict:
    """
    Convenience wrapper: predict with model then evaluate.
    Works with any model that has a .predict() method (XGBoost, sklearn, etc).
    """
    y_pred = model.predict(X)
    return evaluate_predictions(y, y_pred, target_encoders, target_col, split)
