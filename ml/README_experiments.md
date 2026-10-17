# Experiments

All training runs are tracked in MLflow under the experiment name `sentiment`.

```bash
mlflow ui          # local UI at http://127.0.0.1:5000
```

The local file/SQLite backend (`mlruns/`, `mlflow.db`) is used; both are
git-ignored.

## Runs

| Run | Data | max_features | C | Accuracy | F1 |
| --- | --- | --- | --- | --- | --- |
| `baseline-v1` | `data/train.csv` (v1) | 2000 | 1.0 | 0.8700 | 0.8701 |
| `improved-v2` | `data/train_v2.csv` (v2) | 5000 | 1.0 | **0.8740** | **0.8727** |

Both runs were produced at commit `10732ce` and log:

- `git_sha` — code version
- `dataset_file` + `dataset_md5` — data version (from DVC)
- `max_features`, `C` — hyperparameters
- `accuracy`, `f1` — metrics
- the model as an MLflow artifact and as `ml/model.joblib`

## What changed between the two runs

1. **Data**: v2 (cleaned, de-duplicated, length-filtered) instead of v1.
2. **Vocabulary size**: 2,000 → 5,000 TF-IDF features.

Both changes move in the same direction, so the improvement is attributed to
"cleaner data + a larger vocabulary" rather than to a single factor. A future
run could isolate them.

## Production-worthy run

**`improved-v2`.**

- It is better on both metrics: **+0.40 pp accuracy, +0.26 pp F1**.
- It is trained on the cleaned dataset, so it is less likely to have learned
  from HTML markup or duplicated reviews.
- It is fully reproducible: the run records its commit, dataset md5 and
  hyperparameters.

`baseline-v1` is kept as the reference point that shows the cleaning step was
worth doing.

## Metric choice

We optimise **F1** (and report accuracy alongside it) because:

- The training data is nearly balanced, so accuracy alone is already
  meaningful — but it stops being meaningful the moment the incoming class
  balance drifts, which is common in real review traffic.
- Misclassifying a positive review and misclassifying a negative one cost the
  same for this app, so a balanced metric is appropriate.
- F1 penalises a model that gets very good at only one class, which accuracy
  can hide.

Accuracy is reported so that the two numbers can be compared directly, and F1
is the metric we would use to choose between future candidates.
