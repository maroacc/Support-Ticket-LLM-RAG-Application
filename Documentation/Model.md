# Model Documentation

## Feature selection

Only features that are available at the moment a ticket is created were used. 
Features that are only known after the ticket is resolved (e.g. resolution time, resolution code, assigned agent) were excluded, as including them would cause data leakage and make the model unusable in production.

## XGBoost

### Description
For the traditional ML model I chose XGBoost instead of CatBoost. 
The main reason was that it is faster to train in my local computer. However, in
a real production environment with access to cloud services maybe
CatBoost would have been a better choice because it is optimized for categorical classification
and it is less prone to overfitting.
The best option would have been to compare the results of both of them
since XGBoost also allows for more fine-tuning. 

Since XGBoost doesn't allow natively multiclass prediction, I thought it could be a
a good idea to have 2 XGBoost models : one that predicts the category, and a second one that uses
the predictions of the first one to predict the subcategory. Since there are only 5 categories
and knowing the category limits vastly the amount of subcategories possible, I thought this would
help the model indentify the subcategory easily.

### Performance benchmark

The category model achieves 100% weighted F1 on the test set. 
However, the subcategory model only reaches ~20% weighted F1, 
with every individual subcategory scoring around 20%. Since each category has exactly 5 subcategories, 20% is exactly random chance (1/5).

### Feature importance

If we look at teh feature importance, the most for the category are : ____. 
Which is surprising because at first glance they do not seem semantically related to
the category. 

### Error analysis

The difference between the performance at training and test
suggest that the model is overfit for training. Looking at the
confusion ma



### Subcategory is not predictable

After investigating the poor subcategory performance, I conducted an 
thorough analysis (see `notebooks/subcategory_predictability.ipynb`) that shows 
the subcategory labels were most likely randomly assigned within each category during
data generation. The evidence:

1. **Text content is identical** across subcategories within a category, the same templates are used regardless of subcategory.
2. **Chi-squared tests** on all structured features show no significant association with subcategory.
4. **Normalized Mutual Information** between all features and subcategory is effectively zero (~0.0001).

This means no model, regardless of architecture, hyperparameters, or feature engineering 
can predict subcategory better than 20%. The information simply does not 
exist in the dataset. 


## Deep Learning

### Note on subcategory prediction

As shown in the analysis above, subcategory labels are not predictable from any feature in this dataset. The DistilBERT model will therefore not be able to predict subcategories either. However, in a real-world scenario where ticket texts actually differ between subcategories, this architecture would be well-suited for the task  text contains the richest semantic signal for fine-grained classification, which is exactly what transformer models excel at.

### Architecture

The model is a dual-input classifier that combines text understanding with structured features:

```
input_ids + attention_mask ──→ DistilBERT ──→ [CLS] ──→ Dense(128) ──┐
                                                                      ├──→ Dense(128) ──→ Category head (5 classes)
structured features ──→ Dense(128) ──→ Dense(128) ─────────────────── ┘         └──→ Subcategory head (25 classes)
```

**Text branch**: Tokenized ticket text (subject + description + error_logs + stack_trace) is passed through DistilBERT. The [CLS] token embedding (768 dimensions) is projected down to 128 dimensions via a dense layer.

**Structured branch**: Tabular features (one-hot encoded, label-encoded, numeric, and binary columns) are processed through a two-layer MLP (128 units each).

**Merge**: Both branches are concatenated (256 dims) and passed through a shared dense layer (128 units) before feeding into two independent classification heads  one for category, one for subcategory.

Dropout (0.3) is applied after every dense layer.

### Design decisions

- **DistilBERT over full BERT**: 40% smaller and 60% faster while retaining ~97% of BERT's performance. For ticket classification, the full model's capacity is not needed.
- **Dual-input (text + structured)**: Ticket text carries semantic meaning, but structured fields like priority, severity, and product provide signal that's hard to extract from text alone.
- **Two output heads instead of two separate models**: Category and subcategory are related tasks. Shared representations let the model learn features useful for both, and training is more efficient than maintaining two pipelines.
- **70/30 loss weighting toward subcategory**: Subcategory is more granular and harder to predict. Giving it more weight focuses optimization on the harder task.
- **Early stopping (patience=3)**: BERT models overfit quickly on small-to-medium datasets. Early stopping prevents that without needing to manually tune the number of epochs.

### Performance benchmark

Given the randomly assigned subcategory labels, the subcategory head is expected to perform at chance level (~20%). The category head should still achieve high accuracy since the structured features carry strong signal for category prediction.


# Experiment Tracking and Model Lineage

## How MLflow Works in This Project

MLflow tracks every training run using a **SQLite backend** (`mlruns.db` at the project root). Configuration lives in `src/mlflow_config.py` (tracking URI + artifact root only  no hardcoded model names).

### Per-model setup

Each model (XGBoost, CatBoost, etc.) defines its own experiment and registered model name:

| Model | Experiment | Registered Model |
|-------|-----------|-----------------|
| XGBoost | `xgboost-ticket-classifier` | `xgboost-ticket-classifier` |
| CatBoost | `catboost-ticket-classifier` | `catboost-ticket-classifier` |

### What gets logged

All models call `src/mlflow_utils.log_and_register()`, which logs:

- **Parameters**  data splits, hyperparameters, preprocessing config
- **Metrics**  accuracy, weighted F1/precision/recall, per-class F1
- **Artifacts**  joblib files (category model, subcategory model, feature encoders, target encoders)
- **Model version**  auto-registered in the MLflow model registry

### Production promotion

Each model has a `register_production.py` script that sets the `"production"` alias on a model version. The API (`src/api/predict.py`) loads whichever model name it's given by looking up that alias:

```
python -m src.xgboost.register_production   # promote xgboost
python -m src.catboost.register_production   # promote catboost
```

### Viewing results

```
mlflow ui --backend-store-uri sqlite:///mlruns.db
```
