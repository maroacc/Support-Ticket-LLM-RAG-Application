# Model Documentation

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

This shows the metrics for the run with the highest weighted F1 score for the subcategories. 
As it can be seen on the table, the xgboost model for classifying the category performs really well
with a weighted F1 score of almost one on the validation set. However, that is not he case for the subcategory model,
where the weighted F1 score is only 20%. If we look at it in detail, the F1 score for all the subcategories is very similar
to 20%. 20% is also the random probability of choosing the right subcategory given that we know the category. So at first
glance it looks like the model is not learning the subcategory, only choosing it at random from the ones corresponding to the category. 


ADD TABLE


### Feature importance

If we look at teh feature importance, the most for the category are : ____. 
Which is surprising because at first glance they do not seem semantically related to
the category. 

### Error analysis

## Deep Learning

### Description

                                                                                                                                                                                   
  Why DistilBERT (not full BERT)?
  - 40% smaller and 60% faster than BERT-base while retaining ~97% of its performance. For ticket classification, that's a good trade-off — you don't need the full model's         
  capacity.                                                                                                                                                                         
                                                                                                                                                                                    
  Why dual-input (text + structured)?                                                                                                                                               
  - Ticket text carries the semantic meaning of the issue, but structured fields like priority, severity, customer_tier, and product provide signal that's hard to extract from text
   alone. Combining both gives the model more to work with than either branch alone.

  Why fine-tune rather than freeze BERT?
  - Support ticket language (error logs, stack traces, technical jargon) differs from BERT's general pretraining corpus. Fine-tuning lets the encoder adapt to your domain.

  Why two output heads instead of two separate models?
  - Category and subcategory are related tasks. Shared representations in the merged layer let the model learn features useful for both, and training is more efficient than
  maintaining two pipelines.

  Why 70/30 loss weighting toward subcategory?
  - Subcategory is more granular and harder to predict. Giving it more weight in the loss focuses optimization on the harder task. Category likely benefits enough from the shared
  representation.

  Why early stopping?
  - BERT models overfit quickly on small-to-medium datasets. Early stopping with patience=3 prevents that without needing to guess the right number of epochs.


### Performance benchmark

### Feature importance

### Error analysis


# Experiment Tracking and Model Lineage

## How MLflow Works in This Project

MLflow tracks every training run using a **SQLite backend** (`mlruns.db` at the project root). Configuration lives in `src/mlflow_config.py` (tracking URI + artifact root only — no hardcoded model names).

### Per-model setup

Each model (XGBoost, CatBoost, etc.) defines its own experiment and registered model name:

| Model | Experiment | Registered Model |
|-------|-----------|-----------------|
| XGBoost | `xgboost-ticket-classifier` | `xgboost-ticket-classifier` |
| CatBoost | `catboost-ticket-classifier` | `catboost-ticket-classifier` |

### What gets logged

All models call `src/mlflow_utils.log_and_register()`, which logs:

- **Parameters** — data splits, hyperparameters, preprocessing config
- **Metrics** — accuracy, weighted F1/precision/recall, per-class F1
- **Artifacts** — joblib files (category model, subcategory model, feature encoders, target encoders)
- **Model version** — auto-registered in the MLflow model registry

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
