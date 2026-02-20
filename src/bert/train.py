"""
DistilBERT dual-input ticket classifier: category + subcategory.

Fine-tunes DistilBERT on concatenated text fields and combines with
structured features via an MLP, producing two classification heads.

Usage:
    python -m src.bert.train
"""
import sys
import joblib
import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# add parent (src/) to path so we can import shared modules
sys.path.append(str(Path(__file__).resolve().parent.parent))

from data_utils import load_data, DATA_PATH, RANDOM_SEED
from evaluate import evaluate_predictions, plot_confusion_matrix
from mlflow_utils import log_and_register
from preprocessing import preprocess, MAX_LENGTH, MODEL_NAME as BERT_MODEL_NAME
from model import BertTicketClassifier, DENSE_UNITS, DROPOUT_RATE


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT    = Path(__file__).resolve().parent.parent.parent
MODELS_DIR      = PROJECT_ROOT / "models" / "bert"
EXPERIMENT_NAME = "bert-ticket-classifier"
MLF_MODEL_NAME  = "bert-ticket-classifier"

BATCH_SIZE      = 32
EPOCHS          = 10
PATIENCE        = 3         # early stopping: stop if val loss doesn't improve for this many epochs
LEARNING_RATE   = 2e-5      # standard fine-tuning LR for BERT-family models (small to avoid catastrophic forgetting)
CAT_LOSS_W      = 0.3       # weight for category loss in the combined objective
SUBCAT_LOSS_W   = 0.7       # weight for subcategory loss (higher because subcategory is harder/more granular)

# Automatically use GPU if available, otherwise fall back to CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# SPLIT DATA
# ============================================================

def split_data(input_ids, attention_mask, structured, y_cat, y_sub):
    """
    Stratified split of all arrays into train/val/test sets (70/15/15).

    Uses index-based splitting so that all parallel arrays (input_ids,
    attention_mask, structured features, targets) stay aligned.
    Stratification is done on subcategory labels to preserve class
    distribution across splits.
    """
    indices = np.arange(len(input_ids))

    # First split: 70% train, 30% temp (will be split again below)
    idx_train, idx_temp = train_test_split(
        indices, test_size=0.30, random_state=RANDOM_SEED, stratify=y_sub
    )

    # Second split: split the 30% temp equally → 15% validation, 15% test
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=0.50, random_state=RANDOM_SEED, stratify=y_sub[idx_temp]
    )

    # Package each split into a dict for easy downstream access
    splits = {}
    for name, idx in [("train", idx_train), ("val", idx_val), ("test", idx_test)]:
        splits[name] = {
            "input_ids": input_ids[idx],
            "attention_mask": attention_mask[idx],
            "structured": structured[idx],
            "y_cat": y_cat[idx],
            "y_sub": y_sub[idx],
        }

    print(f"Data split complete")
    print(f"   Train      : {len(idx_train)} samples (70%)")
    print(f"   Validation : {len(idx_val)} samples (15%)")
    print(f"   Test       : {len(idx_test)} samples (15%)")

    return splits


# ============================================================
# DATALOADER HELPER
# ============================================================

def make_dataloader(split_data, batch_size, shuffle=True):
    """
    Wrap numpy arrays from a split dict into a PyTorch DataLoader.

    Converts each array to an appropriate tensor dtype:
      - input_ids / attention_mask / targets → long (int64)
      - structured features → float32
    """
    dataset = TensorDataset(
        torch.tensor(split_data["input_ids"], dtype=torch.long),
        torch.tensor(split_data["attention_mask"], dtype=torch.long),
        torch.tensor(split_data["structured"], dtype=torch.float32),
        torch.tensor(split_data["y_cat"], dtype=torch.long),
        torch.tensor(split_data["y_sub"], dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(model, dataloader, optimizer, cat_criterion, sub_criterion):
    """
    Run one full pass over the training data, updating model weights.

    The total loss is a weighted sum of the category and subcategory losses
    (controlled by CAT_LOSS_W and SUBCAT_LOSS_W). Returns the average loss
    per sample across the epoch.
    """
    model.train()  # enable dropout and batch-norm training behavior
    total_loss = 0.0

    for input_ids, attention_mask, structured, y_cat, y_sub in dataloader:
        # Move all tensors to the target device (GPU/CPU)
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        structured = structured.to(DEVICE)
        y_cat = y_cat.to(DEVICE)
        y_sub = y_sub.to(DEVICE)

        # Zero out gradients from the previous step
        optimizer.zero_grad()

        # Forward pass: get raw logits from both classification heads
        cat_logits, sub_logits = model(input_ids, attention_mask, structured)

        # Weighted combination of the two cross-entropy losses
        loss = (CAT_LOSS_W * cat_criterion(cat_logits, y_cat) +
                SUBCAT_LOSS_W * sub_criterion(sub_logits, y_sub))

        # Backprop and parameter update
        loss.backward()
        optimizer.step()

        # Accumulate loss weighted by batch size for correct averaging
        total_loss += loss.item() * input_ids.size(0)

    # Return average loss per sample
    return total_loss / len(dataloader.dataset)


# ============================================================
# VALIDATE
# ============================================================

@torch.no_grad()  # disable gradient computation for efficiency
def validate(model, dataloader, cat_criterion, sub_criterion):
    """
    Compute validation loss without updating weights.

    Same loss calculation as training, but with gradients disabled
    and the model in eval mode (dropout off). Used to monitor
    generalization and trigger early stopping.
    """
    model.eval()  # disable dropout for deterministic evaluation
    total_loss = 0.0

    for input_ids, attention_mask, structured, y_cat, y_sub in dataloader:
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        structured = structured.to(DEVICE)
        y_cat = y_cat.to(DEVICE)
        y_sub = y_sub.to(DEVICE)

        cat_logits, sub_logits = model(input_ids, attention_mask, structured)
        loss = (CAT_LOSS_W * cat_criterion(cat_logits, y_cat) +
                SUBCAT_LOSS_W * sub_criterion(sub_logits, y_sub))

        total_loss += loss.item() * input_ids.size(0)

    return total_loss / len(dataloader.dataset)


# ============================================================
# PREDICT
# ============================================================

@torch.no_grad()
def predict(model, dataloader):
    """
    Generate predictions for all samples in the dataloader.

    Returns two numpy arrays: predicted category indices and predicted
    subcategory indices. Uses argmax over logits to get the most
    likely class for each head.
    """
    model.eval()
    cat_preds, sub_preds = [], []

    for input_ids, attention_mask, structured, _, _ in dataloader:
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        structured = structured.to(DEVICE)

        cat_logits, sub_logits = model(input_ids, attention_mask, structured)

        # Convert logits → predicted class index, move to CPU for numpy
        cat_preds.append(cat_logits.argmax(dim=1).cpu().numpy())
        sub_preds.append(sub_logits.argmax(dim=1).cpu().numpy())

    # Concatenate batches into single arrays
    return np.concatenate(cat_preds), np.concatenate(sub_preds)


# ============================================================
# SAVE ARTIFACTS
# ============================================================

def save_artifacts(model, encoders, target_encoders, version_dir: Path):
    """
    Persist the trained model weights and preprocessing encoders to disk.

    Saved files:
      - model.pt                : PyTorch state dict (model weights)
      - feature_encoders.joblib : fitted tokenizer, label encoders, scaler, etc.
      - target_encoders.joblib  : fitted LabelEncoders for category & subcategory
    """
    version_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), version_dir / "model.pt")
    joblib.dump(encoders, version_dir / "feature_encoders.joblib")
    joblib.dump(target_encoders, version_dir / "target_encoders.joblib")

    print(f"\n Artifacts saved to {version_dir}/")


# ============================================================
# BUILD MLFLOW PARAMS AND METRICS
# ============================================================

def build_params(model, structured_shape, splits, target_encoders, total_samples,
                 best_epoch):
    """
    Collect all hyperparameters and metadata into a flat dict for MLflow logging.
    """
    return {
        # Data info
        "data_source": DATA_PATH,
        "total_samples": total_samples,
        "train_samples": len(splits["train"]["y_cat"]),
        "val_samples": len(splits["val"]["y_cat"]),
        "test_samples": len(splits["test"]["y_cat"]),
        "random_seed": RANDOM_SEED,
        # Model architecture
        "bert_model": BERT_MODEL_NAME,
        "max_length": MAX_LENGTH,
        "dense_units": DENSE_UNITS,
        "dropout_rate": DROPOUT_RATE,
        "num_structured_features": structured_shape,
        "num_categories": len(target_encoders["category"].classes_),
        "num_subcategories": len(target_encoders["subcategory"].classes_),
        # Training hyperparameters
        "batch_size": BATCH_SIZE,
        "max_epochs": EPOCHS,
        "best_epoch": best_epoch,
        "patience": PATIENCE,
        "learning_rate": LEARNING_RATE,
        "cat_loss_weight": CAT_LOSS_W,
        "subcat_loss_weight": SUBCAT_LOSS_W,
        "device": str(DEVICE),
    }


def build_metrics(cat_test_metrics, sub_test_metrics,
                  cat_train_metrics, sub_train_metrics):
    """
    Flatten sklearn classification reports into a single dict of MLflow metrics.

    Extracts weighted-average F1/precision/recall, accuracy, and per-class F1
    for each combination of (category/subcategory) x (train/test).
    """
    metrics = {}

    for prefix, report in [("cat_test", cat_test_metrics),
                           ("cat_train", cat_train_metrics),
                           ("sub_test", sub_test_metrics),
                           ("sub_train", sub_train_metrics)]:
        # Aggregate metrics
        metrics[f"{prefix}_weighted_f1"] = report["weighted avg"]["f1-score"]
        metrics[f"{prefix}_weighted_precision"] = report["weighted avg"]["precision"]
        metrics[f"{prefix}_weighted_recall"] = report["weighted avg"]["recall"]
        metrics[f"{prefix}_accuracy"] = report["accuracy"]

        # Per-class F1 scores (skip summary rows)
        for class_name, class_data in report.items():
            if class_name in ("accuracy", "macro avg", "weighted avg"):
                continue
            metrics[f"{prefix}_f1_{class_name}"] = class_data["f1-score"]

    return metrics


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # 1  Load raw data and run full preprocessing pipeline
    #     (tokenize text, encode features, scale numerics)
    df = load_data(DATA_PATH)
    data = preprocess(df)

    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]
    structured = data["structured"]
    y_cat = data["y_category"]
    y_sub = data["y_subcategory"]
    encoders = data["encoders"]
    target_encoders = data["target_encoders"]

    num_categories = len(target_encoders["category"].classes_)
    num_subcategories = len(target_encoders["subcategory"].classes_)
    total_samples = len(input_ids)

    # 2  Stratified train/val/test split (70/15/15)
    splits = split_data(input_ids, attention_mask, structured, y_cat, y_sub)

    # 3  Create PyTorch DataLoaders for batched iteration
    #     (shuffle training data; keep val/test in order for reproducible evaluation)
    train_loader = make_dataloader(splits["train"], BATCH_SIZE, shuffle=True)
    val_loader = make_dataloader(splits["val"], BATCH_SIZE, shuffle=False)
    test_loader = make_dataloader(splits["test"], BATCH_SIZE, shuffle=False)

    # 4  Instantiate the dual-input model and move to GPU/CPU
    model = BertTicketClassifier(
        num_structured_features=structured.shape[1],
        num_categories=num_categories,
        num_subcategories=num_subcategories,
    ).to(DEVICE)

    print(f"\nModel on device: {DEVICE}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # 5  Set up loss functions and optimizer
    #     CrossEntropyLoss combines LogSoftmax + NLLLoss in one step
    #     AdamW adds weight decay (L2 regularization) decoupled from the gradient update
    cat_criterion = nn.CrossEntropyLoss()
    sub_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # 6  Training loop with early stopping
    #     Track the best validation loss; if it doesn't improve for PATIENCE
    #     consecutive epochs, stop and restore the best model weights
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    best_state = None  # will store a CPU copy of the best model state dict

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer,
                                     cat_criterion, sub_criterion)
        val_loss = validate(model, val_loader, cat_criterion, sub_criterion)

        print(f"  Epoch {epoch:>2}/{EPOCHS}  |  train_loss: {train_loss:.4f}  |  val_loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            # New best  save model state and reset patience
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            # Clone weights to CPU so we don't lose them if training continues on GPU
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            # No improvement  increment patience counter
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch} (best epoch: {best_epoch})")
                break

    # Restore the best model weights before evaluation
    model.load_state_dict(best_state)
    model.to(DEVICE)

    # 7  Evaluate on test and train sets using the best model
    print("\n" + "=" * 60)
    print("  CATEGORY EVALUATION")
    print("=" * 60)

    # Get predictions for both test and train splits
    cat_test_preds, sub_test_preds = predict(model, test_loader)
    cat_train_preds, sub_train_preds = predict(model, train_loader)

    # Category metrics (test and train  train is useful for detecting overfitting)
    cat_test_metrics = evaluate_predictions(
        splits["test"]["y_cat"], cat_test_preds,
        target_encoders, target_col="category", split="test"
    )
    cat_train_metrics = evaluate_predictions(
        splits["train"]["y_cat"], cat_train_preds,
        target_encoders, target_col="category", split="train"
    )

    print("\n" + "=" * 60)
    print("  SUBCATEGORY EVALUATION")
    print("=" * 60)

    # Subcategory metrics
    sub_test_metrics = evaluate_predictions(
        splits["test"]["y_sub"], sub_test_preds,
        target_encoders, target_col="subcategory", split="test"
    )
    sub_train_metrics = evaluate_predictions(
        splits["train"]["y_sub"], sub_train_preds,
        target_encoders, target_col="subcategory", split="train"
    )

    # 8  Plot confusion matrices for all evaluation combinations
    plots_dir = MODELS_DIR / "latest_plots"
    subcat_to_cat = dict(zip(df["subcategory"], df["category"]))
    for target_col, y_true, y_pred, split in [
        ("category",    splits["test"]["y_cat"],  cat_test_preds,  "test"),
        ("category",    splits["train"]["y_cat"], cat_train_preds, "train"),
        ("subcategory", splits["test"]["y_sub"],  sub_test_preds,  "test"),
        ("subcategory", splits["train"]["y_sub"], sub_train_preds, "train"),
    ]:
        plot_confusion_matrix(
            y_true, y_pred, target_encoders,
            target_col=target_col, split=split, save_dir=plots_dir,
            subcat_to_cat=subcat_to_cat if target_col == "subcategory" else None,
        )

    # 9  Save model weights and encoder artifacts to disk
    version_dir = MODELS_DIR / "latest"
    save_artifacts(model, encoders, target_encoders, version_dir)

    # 9  Log everything to MLflow and register the model
    params = build_params(model, structured.shape[1], splits, target_encoders,
                          total_samples, best_epoch)
    metrics = build_metrics(cat_test_metrics, sub_test_metrics,
                            cat_train_metrics, sub_train_metrics)

    run_id, version = log_and_register(
        experiment_name=EXPERIMENT_NAME,
        model_name=MLF_MODEL_NAME,
        run_name="distilbert_dual_input",
        params=params,
        metrics=metrics,
        artifact_dir=version_dir,
        plots_dir=plots_dir,
    )

    print(f"\n   Best epoch: {best_epoch}")
    print(f"   Best val loss: {best_val_loss:.4f}")