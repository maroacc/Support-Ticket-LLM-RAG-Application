import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# ============================================================
# CONFIGURATION
# ============================================================

# Columns to one-hot encode (low cardinality, no natural order)
ONE_HOT_COLS = [
    "customer_tier",
    "channel",
    "environment",
    "language",
    "region",
]

# Columns to label encode (higher cardinality or ordinal-ish)
LABEL_ENCODE_COLS = [
    "product",
    "product_module",
    "product_version",
    "priority",
    "severity",
    "business_impact",
    "customer_sentiment",
]

# Numeric columns — used as-is, no transformation needed for XGBoost
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

# Binary columns — already 0/1 or boolean, just cast to int
BINARY_COLS = [
    "contains_error_code",
    "contains_stack_trace",
    "known_issue",
    "weekend_ticket",
    "after_hours",
]

# Target columns — what we want to predict
TARGET_COLS = ["category", "subcategory"]

# How many of the most frequent tags to use for multi-hot encoding
TOP_N_TAGS = 20

# TF-IDF configuration
TEXT_COLS = ["subject", "description", "error_logs", "stack_trace"]
TFIDF_MAX_FEATURES = 300


# ============================================================
# STEP 1 — ONE-HOT ENCODING
# ============================================================

def encode_one_hot(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode low cardinality categorical columns.
    pd.get_dummies creates one binary column per unique value.
    e.g. channel=portal → channel_portal=1, channel_email=0, ...
    """
    df = pd.get_dummies(df, columns=ONE_HOT_COLS, dtype=int)
    return df


# ============================================================
# STEP 2 — LABEL ENCODING
# ============================================================

def encode_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Label encode higher cardinality categorical columns.
    Each unique string value is mapped to an integer.
    e.g. product_module: encryption_layer→0, backup_scheduler→1, ...

    Returns the modified dataframe AND a dict of fitted encoders
    (needed later to transform new/unseen data the same way).
    """
    encoders = {}

    for col in LABEL_ENCODE_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le  # save encoder so we can reuse it on test data

    return df, encoders


# ============================================================
# STEP 3 — BINARY COLUMNS
# ============================================================

def encode_binary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast boolean/binary columns to integers (True→1, False→0).
    XGBoost expects numbers, not Python booleans.
    """
    for col in BINARY_COLS:
        df[col] = df[col].astype(int)
    return df


# ============================================================
# STEP 4 — MULTI-HOT ENCODING FOR TAGS
# ============================================================

def encode_tags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tags is a list per ticket e.g. ["error", "api", "timeout"].
    We find the TOP_N_TAGS most frequent tags across all tickets,
    then create one binary column per tag.
    e.g. tag_error=1, tag_api=1, tag_timeout=0, ...

    Tickets with no tags get all zeros.
    """
    # Flatten all tags from all tickets into one big list
    all_tags = [tag for tags in df["tags"] for tag in tags]

    # Count frequency of each tag and take the top N
    top_tags = (
        pd.Series(all_tags)
        .value_counts()
        .head(TOP_N_TAGS)
        .index
        .tolist()
    )

    # For each top tag, create a binary column: 1 if ticket has it, 0 otherwise
    for tag in top_tags:
        df[f"tag_{tag}"] = df["tags"].apply(lambda tags: int(tag in tags))

    # Drop the original tags column (can't feed a list into XGBoost)
    df = df.drop(columns=["tags"])

    return df


# ============================================================
# STEP 5 — ENCODE TARGETS
# ============================================================

def encode_targets(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Label encode the target columns (category and subcategory).
    We save the encoders so we can decode predictions back to
    human-readable strings later.
    """
    target_encoders = {}

    for col in TARGET_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        target_encoders[col] = le

    return df, target_encoders


# ============================================================
# STEP 6 — TF-IDF TEXT FEATURES
# ============================================================

def encode_text_tfidf(df: pd.DataFrame) -> tuple[pd.DataFrame, TfidfVectorizer]:
    """
    Combine subject + description into a single text column,
    fit a TF-IDF vectorizer, and append the resulting features
    as tfidf_0, tfidf_1, ... columns.

    Returns the modified dataframe and the fitted vectorizer.
    """
    text = pd.Series("", index=df.index)
    for col in TEXT_COLS:
        text = text + " " + df[col].fillna("").astype(str)
    text = text.str.strip()

    vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(text)

    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=[f"tfidf_{w}" for w in vectorizer.get_feature_names_out()],
        index=df.index,
    )

    df = pd.concat([df, tfidf_df], axis=1)
    df = df.drop(columns=TEXT_COLS)

    return df, vectorizer


# ============================================================
# STEP 7 — DROP COLUMNS WE DON'T NEED
# ============================================================

def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that are:
    - Unique identifiers (no predictive value)
    - Not available at prediction time (post-creation fields)
    - Already processed into other features (e.g. tags)
    """
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

        # Free text — handled by TF-IDF when enabled
        "feedback_text",

        # Datetime — skipping for now
        "created_at",
    ]

    # Only drop columns that actually exist in the dataframe
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    return df


# ============================================================
# MAIN PIPELINE — ties all steps together
# ============================================================

def preprocess(df: pd.DataFrame, use_tfidf: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """
    Full preprocessing pipeline. Returns:
    - X : feature matrix (ready for XGBoost)
    - y : target dataframe with encoded category + subcategory
    - feature_encoders : fitted encoders for categorical features
    - target_encoders  : fitted encoders for targets (to decode predictions)
    """

    df = df.copy()  # don't modify the original dataframe

    # Step 1 — drop columns we don't use
    df = drop_unused_columns(df)

    # Step 2 — TF-IDF text features (or drop text columns if disabled)
    if use_tfidf:
        df, tfidf_vectorizer = encode_text_tfidf(df)
    else:
        df = df.drop(columns=[c for c in TEXT_COLS if c in df.columns])

    # Step 3 — one-hot encode low cardinality categoricals
    df = encode_one_hot(df)

    # Step 4 — label encode higher cardinality categoricals
    df, feature_encoders = encode_labels(df)

    if use_tfidf:
        feature_encoders["tfidf_vectorizer"] = tfidf_vectorizer

    # Step 5 — cast binary columns to int
    df = encode_binary(df)

    # Step 6 — multi-hot encode tags
    df = encode_tags(df)

    # Step 7 — encode targets and separate them from features
    df, target_encoders = encode_targets(df)

    # Split into features (X) and targets (y)
    y = df[TARGET_COLS]
    X = df.drop(columns=TARGET_COLS)

    print(f"✅ Preprocessing complete")
    print(f"   Features shape : {X.shape}")
    print(f"   Targets shape  : {y.shape}")
    print(f"   Feature columns: {list(X.columns)}")

    return X, y, feature_encoders, target_encoders


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    # Load your dataset (expects a CSV or JSON with one ticket per row)
    # df = pd.read_csv("tickets.csv")
    # df = pd.read_json("tickets.json")

    # For a quick test with the single example ticket:
    import json

    sample = {
        "ticket_id": "TK-2024-000001",
        "created_at": "2023-11-02T12:30:10Z",
        "updated_at": "2023-11-02T15:30:46Z",
        "customer_id": "CUST-02387",
        "customer_tier": "starter",
        "organization_id": "ORG-234",
        "product": "CloudBackup Enterprise",
        "product_version": "4.5.10",
        "product_module": "encryption_layer",
        "category": "Feature Request",
        "subcategory": "Documentation",
        "priority": "critical",
        "severity": "P2",
        "channel": "portal",
        "subject": "Request: Add bulk operation support to CloudBackup Enterprise",
        "description": "We would like to request a feature...",
        "error_logs": "",
        "stack_trace": "",
        "customer_sentiment": "frustrated",
        "previous_tickets": 9,
        "resolution": "Issue resolved by updating configuration settings.",
        "resolution_code": "PATCH_APPLIED",
        "resolved_at": "2023-11-02T15:30:46Z",
        "resolution_time_hours": 3.01,
        "resolution_attempts": 3,
        "agent_id": "AGENT-044",
        "agent_experience_months": 41,
        "agent_specialization": "performance",
        "agent_actions": ["consulted_kb", "contacted_customer"],
        "escalated": True,
        "escalation_reason": "SLA breach risk",
        "transferred_count": 0,
        "satisfaction_score": 4,
        "feedback_text": "Bounced between 2 different agents",
        "resolution_helpful": True,
        "tags": ["error", "api", "integration", "timeout", "bug"],
        "related_tickets": [],
        "kb_articles_viewed": ["KB-0218"],
        "kb_articles_helpful": ["KB-0218"],
        "environment": "production",
        "account_age_days": 696,
        "account_monthly_value": 127,
        "similar_issues_last_30_days": 130,
        "product_version_age_days": 24,
        "known_issue": False,
        "bug_report_filed": False,
        "resolution_template_used": "TEMPLATE-AUTH-OPTIMIZE",
        "auto_suggested_solutions": ["KB-1357", "KB-0397"],
        "auto_suggestion_accepted": False,
        "ticket_text_length": 230,
        "response_count": 9,
        "attachments_count": 4,
        "contains_error_code": False,
        "contains_stack_trace": False,
        "business_impact": "high",
        "affected_users": 222,
        "weekend_ticket": False,
        "after_hours": False,
        "language": "de",
        "region": "APAC",
    }

    df = pd.DataFrame([sample])
    X, y, feature_encoders, target_encoders = preprocess(df)