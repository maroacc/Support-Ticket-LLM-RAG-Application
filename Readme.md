# Intelligent Ticket Support System

An end-to-end system that classifies incoming support tickets and retrieves similar past tickets with suggested resolutions.



## Table of Contents

1. [Running the System (Docker)](#running-the-system-docker)
2. [Running the System (Local)](#running-the-system-local)
3. [API Documentation](#api-documentation)
4. [Reproducing Training and Evaluation](#reproducing-training-and-evaluation)
5. [Key Design Decisions and Trade-offs](#key-design-decisions-and-trade-offs)


## Running the System (Docker)

Docker is the recommended way to run the system. It requires [Docker Desktop](https://www.docker.com/products/docker-desktop/). On Windows, use the WSL2 backend (default on Windows 11).

All commands must be run from the **project root** (the folder that contains `Dockerfile` and `src/`).

**Place the dataset** at:
```
data/support_tickets.json
```

### Step 1 : Build RAG artifacts

Pre-compute the ticket embeddings and knowledge graph used by the retrieval system. This only needs to be run once (or when the dataset changes). Requires Python 3.10+ and the dependencies below installed locally.

```bash
# Install dependencies
pip install fastapi uvicorn xgboost catboost scikit-learn pandas numpy \
            sentence-transformers transformers torch mlflow joblib pydantic

# Build the embedding matrix (~160MB, stored in data/rag/embeddings.npy)
python src/rag/build_embeddings.py

# Build the knowledge graph (stored in data/rag/knowledge_graph.json)
python src/rag/build_knowledge_graph.py

# Build resolution stats (stored in data/rag/resolution_stats.json)
python src/rag/build_resolution_stats.py
```

### Step 2 : Train the XGBoost model

```bash
python src/xgboost/train.py
```

This trains only the **category** classifier. Training takes a few minutes on CPU. Artifacts are saved to `models/xgboost/latest/` and logged to MLflow.

### Step 3 : Register the model as production

```bash
python -m src.xgboost.register_production
```

This sets the `production` alias in the MLflow registry. The API loads whichever model carries this alias at startup.

### Step 4 : Build and run the container

```bash
# Build the image (once, or after code changes)
docker build -t ticket-classifier .
```

Mac / Linux:
```bash
docker run -d --name ticket-classifier -p 8000:8000 \
  -v "$(pwd)/mlruns.db:/app/mlruns.db" \
  -v "$(pwd)/mlruns:/app/mlruns" \
  -v "$(pwd)/mlartifacts:/app/mlartifacts" \
  -v "$(pwd)/data/support_tickets.json:/app/data/support_tickets.json:ro" \
  -v "$(pwd)/data/rag:/app/data/rag:ro" \
  -v "$(pwd)/models/xgboost/latest:/app/models/xgboost/latest" \
  -v "hf_cache:/cache/huggingface" \
  ticket-classifier
```

Windows (PowerShell):
```powershell
docker run -d --name ticket-classifier -p 8000:8000 `
  -v "${PWD}/mlruns.db:/app/mlruns.db" `
  -v "${PWD}/mlruns:/app/mlruns" `
  -v "${PWD}/mlartifacts:/app/mlartifacts" `
  -v "${PWD}/data/support_tickets.json:/app/data/support_tickets.json:ro" `
  -v "${PWD}/data/rag:/app/data/rag:ro" `
  -v "${PWD}/models/xgboost/latest:/app/models/xgboost/latest" `
  -v "hf_cache:/cache/huggingface" `
  ticket-classifier
```

> **Note**: if `${PWD}` does not resolve correctly, replace it with the absolute path to the project root (e.g. `C:/Users/Maria/PycharmProjects/full_stack_ai_callenge`).

To stop and restart the container:
```bash
docker stop ticket-classifier && docker rm ticket-classifier
```

Volume breakdown:
| Volume | Purpose |
|||
| `mlruns.db` | MLflow tracking database : persists experiment history |
| `mlruns/` | MLflow artifact files (model weights, encoders) referenced by the database |
| `mlartifacts/` | MLflow model artifacts : persists trained model files |
| `data/support_tickets.json` | Raw dataset for `/train` (read-only) |
| `data/rag/` | Pre-built RAG artifacts from Step 1 (read-only) |
| `models/xgboost/latest/` | Local model copy written by `train.py` |
| `hf_cache` | sentence-transformers model cache : downloaded once, reused across restarts |

The API is now available at http://127.0.0.1:8000. Interactive docs at http://localhost:8000/docs.



## Running the System (Local)

To run without Docker, follow steps 1–3 above, then start the API directly:

```bash
python -m uvicorn src.api.app:app --port 8000
```



## API Documentation

### POST /predict

Classifies a support ticket using the XGBoost classifier. 

#### Request

_Required fields :_ subject, description, error_logs, product, product_version 
, product_module, priority, severity, customer_tier, channel, environment


Example : 
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

#### Response
```json
{
  "category": "Technical Issue",
  "subcategory": null
}
```

`subcategory` is always `null` : see [On Subcategory Prediction](#on-subcategory-prediction).



### POST /rag

Finds similar historical tickets that have been solved and returns their information and solutions.

Call `/predict` first to get the category, then pass it along with the ticket fields in this request. The category is used as an additional matching signal by the retrieval system.

#### Request :

_Parameters :_
- top_k (optional) : number of results to return. Default 5

_Required fields :_ subject, description, error_logs, product, product_version, product_module, category

Example
```json
{
  "subject": "Database sync failing with timeout error",
  "description": "Getting ERROR_TIMEOUT_429 when syncing large datasets.",
  "error_logs": "ERROR_TIMEOUT_429: Connection timeout after 30s",
  "product": "DataSync Pro",
  "product_version": "3.2.1",
  "product_module": "sync_engine",
  "category": "Technical Issue"
}
```

#### Response
```json
{
  "category": "Technical Issue",
  "results": [
    {
      "ticket_id": "TK-2024-001234",
      "subject": "Database sync failing with timeout error",
      "similarity_score": 0.9721,
      "match_ratio": 0.8,
      "final_score": 0.9033,
      "solution": {
        "resolution": "Increased batch size limits in config.yaml...",
        "resolution_code": "CONFIG_CHANGE",
        "resolution_template": "TEMPLATE-DB-TIMEOUT",
        "resolution_helpful": true,
        "kb_articles": [
          { "article": "KB-887", "success_rate": 0.85 },
          { "article": "KB-429", "success_rate": 0.61 }
        ]
      }
    }
  ]
}
```

`category` is echoed back from the request to confirm what was used for matching.

Results are sorted by final_score (60% semantic similarity + 40% knowledge graph field overlap).

Each result includes the KB articles that helped resolve that ticket, along with their historical success rates.

> **Evolution**: the current response returns one solution per similar ticket. A natural next step would be to aggregate the KB articles across all returned tickets into a single deduplicated list ordered by success rate, surfacing the most effective solutions first regardless of which ticket they came from.

The endpoint returns 503 if the RAG artifacts haven't been built yet.

#### POST /train

Triggers a training run for the specified model. The request blocks until training completes.

> **Note**: This has been implemented as a synchronous endpoint, but ideally it should be asynchronous with a job status endpoint.

**Request**
```json
{
  "model": "xgboost"
}
```

| Field | Type | Default | Description |
|||||
| `model` | string | required | `"xgboost"` |

**Response**
```json
{
  "model_name": "xgboost-ticket-classifier",
  "run_id": "772c82cecc4048dabf0b9796e1ead5ac",
  "version": "11"
}
```

After training, the new version must be manually promoted to production. The reason is to have control
over what is deployed into production.

```bash
python -m src.xgboost.register_production
```
 

## Reproducing Training and Evaluation

The models have seeds and the parameters in the code are set to the benchmark values.

### XGBoost (recommended)

```bash
python src/xgboost/train.py
```

### Scheduled retraining (Airflow)

An Airflow DAG triggers a training run every day at 03:00 UTC by calling **POST /train**. 
After the run completes, the new model version is registered in MLflow but not promoted to production, since that step
is intented to be manual to have control over what is deployed into production.

```bash
python -m src.xgboost.register_production
```


> **Note on retraining**: the DAG retrains from scratch on the full dataset. 
> In production you would instead fine-tune the existing model on recent data 
> only, which is faster and avoids deleting patterns that were previously learnt.

### Viewing results in MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlruns.db
```

Open `http://127.0.0.1:5000` to browse experiment runs, metrics, and confusion matrix plots.

### CatBoost (reference only)

CatBoost is included for comparison. 
It handles categorical features natively without label encoding and may 
generalise better on real-world data where categories are less cleanly separable.
However, for this PoC XGboost was chosen because it was good enough and faster to train.

```bash
python src/catboost/train.py
```

### DistilBERT deep learning model (reference only)

```bash
python src/bert/train.py
```

The deep learning model is not connected to the API or the RAG pipeline. 
Training it on CPU is incredibly slow, and given that XGBoost already 
achieves 99% weighted F1 on category with structured features alone, 
there is no practical justification for using it here. 
It is kept as a reference implementation only.


### Data split

All models use the same 70/15/15 train/validation/test split 
with stratification and a fixed random seed (42) for reproducibility.

| Split      | Size                  |
|------------|-----------------------|
| Train      | 70% (~70,000 tickets) |
| Validation | 15% (~15,000 tickets) |
| Test       | 15% (~15,000 tickets) |


## Key Design Decisions and Trade-offs

### Subcategory prediction

Subcategory labels in this dataset are not predictable. As a result, all models predict only the category. 
The script train_hierarchical.py is kept as a reference showing how a category + subcategory 
pipeline would be structured, but it is not connected to the API 
and the RAG system only uses the category value.

The analysis of predictability can be found at _notebooks/subcategory_predictibility_.

### XGBoost Choice

XGBoost achieves 99% weighted F1 on the category classification
using only structured features (priority, severity, product, customer metadata). 
Given this performance, there is no reason to use a more complex model:

- **CatBoost** reaches the same F1 score and is kept as a reference, but 
it takes longer to train so for this use case XGBoost performs better.
- **DistilBERT** would probably achieve a better performance in a real environment
with real data instead of synthetic data. 
But in this case it adds enormous training and prediction cost for zero additional 
benefit on this dataset.

In reality, for this dataset probably an even lighter model would achieve
similar results because according to the analysis, because in the feature importance
very few features seemed meaningful. However, I did not have enough time to
test that approach.


### RAG retrieval

The retrieval system combines two signals to rank similar historical tickets:

- **Semantic similarity**: computes similarity between ticket texts by calculating
their embeddings and computing the cosine distance between them. This was given a weight
of 60%.
- **Metadata and keyword matching** : exact match on product, product version, module, category, and extracted error codes.
This was given a weight of 40%.

The actual formula is : 

```
    similarity = 0.6*cosine_similarity + 0.4*(same_product + same_product_version + same_product_module + same_category)
```

Since the subcategory cannot be predicted, it was not used for the RAG retrieval as
the predicted result would most likely be wrong and pollute the results.


The embedding matrix (aprox. 160MB for 100K tickets) is loaded entirely into memory 
at startup. This works at the current scale but would need to be replaced 
with a dedicated vector database (p. ex pgvector) as ticket volume 
grows. See `Documentation/Architecture.md` for details.

Each result includes a structured solution extracted from the matched 
historical ticket:

```json
{
  "resolution":          "Increased batch size limits in config.yaml...",
  "resolution_code":     "CONFIG_CHANGE",
  "resolution_template": "TEMPLATE-DB-TIMEOUT",
  "resolution_helpful":  true,
  "kb_articles": [
    { "article": "KB-887", "success_rate": 0.85 },
    { "article": "KB-429", "success_rate": 0.61 }
  ]
}
```

KB articles are extracted from the `kb_articles_helpful` field of 
the matched ticket and enriched with their historical success rates 
from `data/rag/resolution_stats.json` (pre-computed from all 100K tickets), 
then sorted by success rate descending. 



### MLflow tracking

All training runs are logged to a local SQLite backend (`mlruns.db`). 
Models are promoted to production via a `"production"` alias : 
the API resolves this alias at startup. To chang the production 
model, the `register_production` function has to be ran with the 
desired version, with no code changes or API restart.


### Implemented vs. ideal architecture

This codebase uses local files in place 
of production infrastructure 
(data lake, vector database, graph database). 
See `Documentation/Architecture.md` for a full description of what 
the production system would look like and the reasons behind each 
component choice.
