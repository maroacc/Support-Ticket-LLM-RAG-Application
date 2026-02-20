import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from transformers import DistilBertTokenizer


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "distilbert-base-uncased"  # HuggingFace model identifier for tokenizer + encoder
MAX_LENGTH = 256                        # max token length; longer sequences are truncated, shorter are padded

# Text columns  concatenated into a single string per ticket
# These capture the free-text description of the support issue
TEXT_COLS = ["subject", "description", "error_logs", "stack_trace"]

# Columns to one-hot encode (low cardinality  creates one binary column per unique value)
ONE_HOT_COLS = [
    "customer_tier",
    "channel",
    "environment",
    "language",
    "region",
]

# Columns to label encode (higher cardinality  maps each unique value to an integer)
# One-hot encoding these would create too many sparse columns
LABEL_ENCODE_COLS = [
    "product",
    "product_module",
    "product_version",
    "priority",
    "severity",
    "business_impact",
    "customer_sentiment",
]

# Numeric columns  will be standardized (zero mean, unit variance)
# to help the neural network converge faster
NUMERIC_COLS = [
    "previous_tickets",
    "account_age_days",
    "account_monthly_value",
    "similar_issues_last_30_days",
    "product_version_age_days",
    "affected_users",
    "attachments_count",
    "ticket_text_length",
]

# Binary columns  already 0/1 but may be stored as bool; cast to int
BINARY_COLS = [
    "contains_error_code",
    "contains_stack_trace",
    "known_issue",
    "weekend_ticket",
    "after_hours",
]

# Target columns  what the model learns to predict
TARGET_COLS = ["category", "subcategory"]

# Tags  keep only the TOP_N_TAGS most frequent tags as multi-hot features
# to avoid an explosion of sparse columns from rare tags
TOP_N_TAGS = 20

# Columns to drop  either not available at prediction time (e.g. resolution info)
# or are identifiers that shouldn't be used as features
COLS_TO_DROP = [
    "ticket_id", "customer_id", "organization_id", "agent_id",
    "updated_at", "resolution", "resolution_code", "resolved_at",
    "resolution_time_hours", "resolution_attempts", "agent_experience_months",
    "agent_specialization", "agent_actions", "escalated", "escalation_reason",
    "transferred_count", "satisfaction_score", "feedback_text",
    "resolution_helpful", "kb_articles_viewed", "kb_articles_helpful",
    "bug_report_filed", "resolution_template_used", "auto_suggested_solutions",
    "auto_suggestion_accepted", "response_count", "related_tickets",
    "created_at",
]


# ============================================================
# PREPROCESS
# ============================================================

def preprocess(df: pd.DataFrame) -> dict:
    """
    Preprocess raw ticket data for the DistilBERT dual-input model.

    Pipeline steps:
      1. Concatenate text columns and tokenize with DistilBERT tokenizer
      2. Drop columns not useful for prediction
      3. Encode target labels (category, subcategory) as integers
      4. One-hot encode low-cardinality categoricals
      5. Label encode high-cardinality categoricals
      6. Cast binary columns to int
      7. Create multi-hot features from the top-N most frequent tags
      8. Standardize numeric columns (zero mean, unit variance)
      9. Convert all structured features to a single numpy array

    Returns a dict with:
      - input_ids       : tokenized text input IDs (n_samples, MAX_LENGTH)
      - attention_mask   : attention masks for padding (n_samples, MAX_LENGTH)
      - structured       : numpy array of structured features (n_samples, n_features)
      - y_category       : encoded category targets
      - y_subcategory    : encoded subcategory targets
      - encoders         : dict of fitted encoders (label encoders, scaler, top_tags, tokenizer)
      - target_encoders  : dict of fitted target label encoders
    """

    df = df.copy()  # avoid mutating the original DataFrame

    # ---- 1. Extract and tokenize text ----
    # Concatenate all text columns into one string per ticket, separated by spaces.
    # Missing values are replaced with empty strings to avoid "nan" in the text.
    text = pd.Series("", index=df.index)
    for col in TEXT_COLS:
        text = text + " " + df[col].fillna("").astype(str)
    text = text.str.strip().tolist()

    # Tokenize using DistilBERT's WordPiece tokenizer:
    #   - padding="max_length" ensures all sequences have the same length
    #   - truncation=True cuts sequences longer than MAX_LENGTH
    #   - return_tensors="np" returns numpy arrays (converted to PyTorch tensors later)
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    encoded_text = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="np",
    )

    input_ids = encoded_text["input_ids"]          # token indices (vocab lookup)
    attention_mask = encoded_text["attention_mask"]  # 1 for real tokens, 0 for padding

    # ---- 2. Drop unused columns ----
    # Remove columns that leak future info or are just identifiers,
    # and also remove text columns (already tokenized above)
    cols_to_drop = [c for c in COLS_TO_DROP if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    df = df.drop(columns=[c for c in TEXT_COLS if c in df.columns])

    # ---- 3. Encode targets ----
    # Convert string labels to integer indices for CrossEntropyLoss
    target_encoders = {}
    for col in TARGET_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        target_encoders[col] = le  # save to decode predictions back to strings later

    y_category = df["category"].values
    y_subcategory = df["subcategory"].values
    df = df.drop(columns=TARGET_COLS)  # remove targets from feature DataFrame

    # ---- 4. One-hot encode categoricals ----
    # Converts each categorical column into multiple binary columns
    # e.g. customer_tier="gold" → customer_tier_gold=1, customer_tier_silver=0, ...
    df = pd.get_dummies(df, columns=ONE_HOT_COLS, dtype=int)

    # ---- 5. Label encode higher cardinality categoricals ----
    # Maps each unique string value to an integer (0, 1, 2, ...)
    # More compact than one-hot for columns with many unique values
    label_encoders = {}
    for col in LABEL_ENCODE_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # ---- 6. Binary columns ----
    # Ensure boolean-like columns are stored as int (0/1)
    for col in BINARY_COLS:
        df[col] = df[col].astype(int)

    # ---- 7. Multi-hot tags ----
    # Each ticket has a list of tags. We keep only the TOP_N_TAGS most frequent
    # tags and create a binary column for each (1 if the ticket has that tag).
    all_tags = [tag for tags in df["tags"] for tag in tags]
    top_tags = (
        pd.Series(all_tags)
        .value_counts()
        .head(TOP_N_TAGS)
        .index
        .tolist()
    )
    for tag in top_tags:
        df[f"tag_{tag}"] = df["tags"].apply(lambda tags: int(tag in tags))
    df = df.drop(columns=["tags"])  # raw tags list no longer needed

    # ---- 8. Standardize numeric columns ----
    # Scale to zero mean and unit variance so that features with large ranges
    # (e.g. account_age_days) don't dominate the gradient updates
    scaler = StandardScaler()
    df[NUMERIC_COLS] = scaler.fit_transform(df[NUMERIC_COLS])

    # ---- 9. Convert structured features to numpy ----
    # At this point all remaining columns are numeric; stack into a single array
    structured_cols = list(df.columns)
    structured = df.values.astype(np.float32)

    # Bundle all fitted encoders so they can be reused at inference time
    encoders = {
        "tokenizer": tokenizer,
        "label_encoders": label_encoders,
        "scaler": scaler,
        "top_tags": top_tags,
        "structured_columns": structured_cols,
    }

    print(f"Preprocessing complete")
    print(f"   Text tokens shape : {input_ids.shape}")
    print(f"   Structured shape  : {structured.shape}")
    print(f"   Categories        : {len(target_encoders['category'].classes_)}")
    print(f"   Subcategories     : {len(target_encoders['subcategory'].classes_)}")

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "structured": structured,
        "y_category": y_category,
        "y_subcategory": y_subcategory,
        "encoders": encoders,
        "target_encoders": target_encoders,
    }