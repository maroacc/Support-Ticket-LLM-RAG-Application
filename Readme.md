# Intelligent Product Support System

An end-to-end system that classifies incoming support tickets and retrieves similar past tickets with suggested resolutions.

---

## Table of Contents

1. [Setup and Installation](#setup-and-installation)
2. [Running the System](#running-the-system)
3. [API Documentation](#api-documentation)
4. [Reproducing Training and Evaluation](#reproducing-training-and-evaluation)
5. [Key Design Decisions and Trade-offs](#key-design-decisions-and-trade-offs)

---

## Setup and Installation

**Requirements**: Python 3.10+

```bash
# Clone the repo and create a virtual environment
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

# Install dependencies
pip install fastapi uvicorn xgboost catboost scikit-learn pandas numpy \
            sentence-transformers transformers torch mlflow joblib pydantic
```

**Place the dataset** at:
```
data/support_tickets.json
```

---

## Running the System

Follow these steps in order the first time. After the initial setup, only step 4 is needed to start the API.

### Step 1 — Build RAG artifacts

Pre-compute the ticket embeddings and knowledge graph used by the retrieval system. This only needs to be run once (or when the dataset changes).

```bash
# Build the embedding matrix (~160MB, stored in data/rag/embeddings.npy)
python src/rag/build_embeddings.py

# Build the knowledge graph (stored in data/rag/knowledge_graph.json)
python src/rag/build_knowledge_graph.py

# Build resolution stats (stored in data/rag/resolution_stats.json)
python src/rag/build_resolution_stats.py
```

### Step 2 — Train the XGBoost model

```bash
python src/xgboost/train.py
```

This trains only the **category** classifier (see [why below](#on-subcategory-prediction)). Training takes a few minutes on CPU. Artifacts are saved to `models/xgboost/latest/` and logged to MLflow.

### Step 3 — Register the model as production

```bash
python -m src.xgboost.register_production
```

This sets the `production` alias in the MLflow registry. The API loads whichever model carries this alias at startup.

### Step 4 — Start the API

```bash
python -m uvicorn src.api.app:app --port 8000
```

The API is now available at `http://127.0.0.1:8000`. Interactive docs at `http://127.0.0.1:8000/docs`.

---

## API Documentation

### `POST /predict`

Classifies a support ticket. Only `subject` and `description` are required — all other fields default to sensible values.

**Request**
```json
{
  "subject": "Database sync failing with timeout error",
  "description": "Getting ERROR_TIMEOUT_429 when syncing large datasets. Started after the recent update.",
  "error_logs": "ERROR_TIMEOUT_429: Connection timeout after 30s",
  "product": "DataSync Pro",
  "product_version": "3.2.1",
  "product_module": "sync_engine",
  "priority": "high",
  "severity": "P2",
  "customer_tier": "enterprise",
  "channel": "email",
  "environment": "production"
}
```

**Response**
```json
{
  "category": "Technical Issue",
  "subcategory": null
}
```

`subcategory` is always `null` — see [On Subcategory Prediction](#on-subcategory-prediction).

---

### `POST /train`

Triggers a training run for the specified model. The request blocks until training completes.

> **Note**: This is a synchronous endpoint intended for internal use. On large datasets, XGBoost training takes a few minutes. For production use this should be made asynchronous with a job status endpoint.

**Request**
```json
{
  "model": "xgboost"
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `model` | string | required | `"xgboost"` |

**Response**
```json
{
  "model_name": "xgboost-ticket-classifier",
  "run_id": "772c82cecc4048dabf0b9796e1ead5ac",
  "version": "11"
}
```

After training, the new version must be manually promoted to production:
```bash
python -m src.xgboost.register_production
```

---

## Reproducing Training and Evaluation

### XGBoost (recommended)

```bash
python src/xgboost/train.py
```

### Viewing results in MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlruns.db
```

Open `http://127.0.0.1:5000` to browse experiment runs, metrics, and confusion matrix plots.

### CatBoost (reference only)

CatBoost is included for comparison. It handles categorical features natively without label encoding and may generalise better on real-world data where categories are less cleanly separable.

```bash
python src/catboost/train.py
```

### DistilBERT deep learning model (reference only)

```bash
python src/bert/train.py
```

The deep learning model is **not connected to the API or the RAG pipeline**. Training it on CPU is prohibitively slow, and given that XGBoost already achieves ~99% weighted F1 on category with structured features alone, there is no practical justification for using it here. It is kept as a reference implementation only.

---

### Data split

All models use the same 70/15/15 train/validation/test split with stratification and a fixed random seed (`42`) for reproducibility.

| Split | Size |
|---|---|
| Train | 70% (~70,000 tickets) |
| Validation | 15% (~15,000 tickets) |
| Test | 15% (~15,000 tickets) |

---

## Key Design Decisions and Trade-offs

### On subcategory prediction

**Subcategory labels in this dataset are not predictable.** Analysis (see `notebooks/subcategory_predictability.ipynb` and `Documentation/Model.md`) shows they were randomly assigned during data generation — the same ticket text templates are reused across all subcategories within a category, and statistical tests confirm zero mutual information between any feature and subcategory label. Every model tested scores at exactly chance level (~20%, i.e. 1 in 5) regardless of architecture or features.

As a result, **all models predict only category**. `train_hierarchical.py` is kept as a reference showing how a category + subcategory pipeline would be structured, but it is not connected to the API.

---

### Why XGBoost and not a heavier model

XGBoost achieves **~99% weighted F1** on the category classification task using only structured features (priority, severity, product, customer metadata). Given this performance, there is no reason to use a more complex model:

- **CatBoost** reaches the same performance and is kept as a reference showing an alternative approach to categorical feature handling.
- **DistilBERT** is architecturally interesting but adds enormous training cost for zero additional benefit on this dataset.
- Honestly, an even lighter model — logistic regression or a shallow decision tree — would likely achieve the same result, since the structured features carry all the signal needed. The practical takeaway is to match model complexity to the actual difficulty of the task.

---

### RAG retrieval

The retrieval system combines two signals to rank similar historical tickets:

- **Semantic similarity** (60% weight): `sentence-transformers/all-MiniLM-L6-v2` embeddings, searched via dot product on a pre-computed NumPy matrix.
- **Structured field matching** (40% weight): exact match on product, product version, module, category, and extracted error codes.

Subcategory was intentionally excluded from structured matching — since it carries no real information in this dataset, including it would only add noise to the ranking score.

The embedding matrix (~160MB for 100K tickets) is loaded entirely into memory at startup. This works at the current scale but would need to be replaced with a dedicated vector database (pgvector, Qdrant, Pinecone) as ticket volume grows. See `Documentation/Architecture.md` for details.

---

### MLflow tracking

All training runs are logged to a local SQLite backend (`mlruns.db`). Models are promoted to production via a `"production"` alias — the API resolves this alias at startup. Switching the production model requires only re-running `register_production` with the desired version, with no code changes or API restart.

---

### Implemented vs. ideal architecture

This codebase uses local files and in-process storage in place of production infrastructure (data lake, vector database, graph database, orchestration). See `Documentation/Architecture.md` for a full description of what the production system would look like and the rationale behind each component choice.
