# Current production model

| Field | Value |
| --- | --- |
| MLflow run | `improved-v2` (experiment `sentiment`) |
| Git commit | `10732ce` |
| Dataset | `data/train_v2.csv`, DVC md5 `e2a4eedda6029ed680ae015c29483e47` |
| Model artifact | `ml/model.joblib`, DVC md5 `12ae722dd99fcc571c2d190ccc6285a6` |
| Pipeline | TF-IDF (`max_features=5000`) + `LogisticRegression(C=1.0)` |
| Metrics | accuracy **0.8740**, F1 **0.8727** on `data/test.csv` |

`ml/model.joblib` is a **single sklearn `Pipeline`** (TF-IDF + classifier), so
the API can pass raw text straight to it:

```python
model.predict(["This movie was great"])     # -> [1]
```

## Reproduce

```bash
dvc pull
python -m ml.train \
  --train data/train_v2.csv \
  --test data/test.csv \
  --max-features 5000 --C 1.0 \
  --run-name improved-v2
```
