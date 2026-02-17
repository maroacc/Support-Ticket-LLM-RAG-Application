"""
CatBoost-native preprocessing.

Key difference from XGBoost preprocessing:
  - No label encoding for categoricals — CatBoost handles them natively
    via the cat_features parameter (pass categorical columns as strings)
  - No one-hot encoding — CatBoost encodes internally
  - Keep: TF-IDF for text, multi-hot for tags, binary cast, numeric as-is,
    drop unused columns, encode targets
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# CONFIGURATION
# ============================================================

# Categorical columns — passed as strings to CatBoost (no encoding needed)
CAT_COLS = [
    "customer_tier",
    "channel",
    "environment",
    "language",
    "region",
    "product",
    "product_module",
    "product_version",
    "priority",
    "severity",
    "business_impact",
    "customer_sentiment",
]

# Numeric columns — used as-is
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

# Binary columns — cast to int
BINARY_COLS = [
    "contains_error_code",
    "contains_stack_trace",
    "known_issue",
    "weekend_ticket",
    "after_hours",
]

# Target columns
TARGET_COLS = ["category", "subcategory"]

# How many of the most frequent tags to use for multi-hot encoding
TOP_N_TAGS = 20

# TF-IDF configuration
TEXT_COLS = ["subject", "description", "error_logs", "stack_trace"]
TFIDF_MAX_FEATURES = 300


# ============================================================
# STEP 1 — DROP UNUSED COLUMNS
# ============================================================

def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [
        # Identifiers
        "ticket_id", "customer_id", "organization_id", "agent_id",

        # Not available at prediction time
        "updated_at", "resolution", "resolution_code", "resolved_at",
        "resolution_time_hours", "resolution_attempts", "agent_experience_months",
        "agent_specialization", "agent_actions", "escalated", "escalation_reason",
        "transferred_count", "satisfaction_score", "feedback_text",
        "resolution_helpful", "kb_articles_viewed", "kb_articles_helpful",
        "bug_report_filed", "resolution_template_used", "auto_suggested_solutions",
        "auto_suggestion_accepted", "response_count", "related_tickets",

        # Datetime
        "created_at",
    ]

    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    return df


# ============================================================
# STEP 2 — CATEGORICAL COLUMNS (strings for CatBoost)
# ============================================================

def prepare_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure categorical columns are strings (CatBoost handles encoding)."""
    for col in CAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df


# ============================================================
# STEP 3 — BINARY COLUMNS
# ============================================================

def encode_binary(df: pd.DataFrame) -> pd.DataFrame:
    for col in BINARY_COLS:
        df[col] = df[col].astype(int)
    return df


# ============================================================
# STEP 4 — MULTI-HOT ENCODING FOR TAGS
# ============================================================

def encode_tags(df: pd.DataFrame) -> pd.DataFrame:
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

    df = df.drop(columns=["tags"])
    return df


# ============================================================
# STEP 5 — TF-IDF TEXT FEATURES
# ============================================================

def encode_text_tfidf(df: pd.DataFrame) -> tuple[pd.DataFrame, TfidfVectorizer]:
    text = pd.Series("", index=df.index)
    for col in TEXT_COLS:
        text = text + " " + df[col].fillna("").astype(str)
    text = text.str.strip()

    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES, stop_words="english", ngram_range=(1, 2)
    )
    tfidf_matrix = vectorizer.fit_transform(text)

    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=[f"tfidf_{w}" for w in vectorizer.get_feature_names_out()],
        index=df.index,
    )

    df = pd.concat([df, tfidf_df], axis=1)
    df = df.drop(columns=[c for c in TEXT_COLS if c in df.columns])
    return df, vectorizer


# ============================================================
# STEP 6 — ENCODE TARGETS
# ============================================================

def encode_targets(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    target_encoders = {}
    for col in TARGET_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        target_encoders[col] = le
    return df, target_encoders


# ============================================================
# MAIN PIPELINE
# ============================================================

def preprocess(df: pd.DataFrame, use_tfidf: bool = False):
    """
    Full CatBoost preprocessing pipeline. Returns:
    - X : feature DataFrame (categoricals as strings, numerics as numbers)
    - y : target DataFrame with encoded category + subcategory
    - feature_encoders : dict with cat_feature_names, cat_feature_indices,
                         and optionally tfidf_vectorizer
    - target_encoders  : fitted LabelEncoders for targets
    """
    df = df.copy()

    # Step 1 — drop unused columns
    df = drop_unused_columns(df)

    # Step 2 — TF-IDF or drop text columns
    tfidf_vectorizer = None
    if use_tfidf:
        df, tfidf_vectorizer = encode_text_tfidf(df)
    else:
        df = df.drop(columns=[c for c in TEXT_COLS if c in df.columns])

    # Step 3 — ensure categoricals are strings
    df = prepare_categoricals(df)

    # Step 4 — cast binary columns to int
    df = encode_binary(df)

    # Step 5 — multi-hot encode tags
    df = encode_tags(df)

    # Step 6 — encode targets
    df, target_encoders = encode_targets(df)

    # Split features and targets
    y = df[TARGET_COLS]
    X = df.drop(columns=TARGET_COLS)

    # Identify categorical feature indices for CatBoost
    cat_feature_names = [col for col in CAT_COLS if col in X.columns]
    cat_feature_indices = [X.columns.get_loc(col) for col in cat_feature_names]

    # Build feature encoders dict
    feature_encoders = {
        "cat_feature_names": cat_feature_names,
        "cat_feature_indices": cat_feature_indices,
    }
    if tfidf_vectorizer is not None:
        feature_encoders["tfidf_vectorizer"] = tfidf_vectorizer

    print(f"Preprocessing complete (CatBoost)")
    print(f"   Features shape : {X.shape}")
    print(f"   Targets shape  : {y.shape}")
    print(f"   Categorical features ({len(cat_feature_names)}): {cat_feature_names}")
    print(f"   Feature columns: {list(X.columns)}")

    return X, y, feature_encoders, target_encoders
