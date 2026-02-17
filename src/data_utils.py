import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH    = str(PROJECT_ROOT / "data" / "support_tickets.json")
TEST_SIZE   = 0.15                             # 15% for test
VAL_SIZE    = 0.15                             # 15% for validation
RANDOM_SEED = 42                               # for reproducibility


# ============================================================
# LOAD DATA
# ============================================================

def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Load the dataset from a JSON file.
    Expects a list of ticket objects, one per row.
    e.g. [ { "ticket_id": "TK-001", ... }, { ... } ]
    """
    df = pd.read_json(path)
    print(f"✅ Loaded {len(df)} tickets from {path}")
    return df


# ============================================================
# SPLIT DATA
# ============================================================

def split_data(X: pd.DataFrame, y: pd.Series):
    """
    Split into train / validation / test sets (70 / 15 / 15).

    We split in two steps:
      1. Split off 30% as temp set (will become val + test)
      2. Split that 30% in half → 15% val, 15% test
    """

    # step 1 — split into 70% train and 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=(TEST_SIZE + VAL_SIZE),   # 30%
        random_state=RANDOM_SEED,
        stratify=y                          # preserve class distribution in each split
    )

    # step 2 — split temp evenly into validation and test (50/50 of the 30%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,                      # 50% of 30% = 15%
        random_state=RANDOM_SEED,
        stratify=y_temp
    )

    print(f"✅ Data split complete")
    print(f"   Train      : {len(X_train)} samples (70%)")
    print(f"   Validation : {len(X_val)} samples (15%)")
    print(f"   Test       : {len(X_test)} samples (15%)")

    return X_train, X_val, X_test, y_train, y_val, y_test
