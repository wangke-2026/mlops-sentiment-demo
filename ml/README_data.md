# Data

## Source

Raw data: [IMDB-Movie-Reviews-Large-Dataset-50k](https://github.com/laxmimerit/IMDB-Movie-Reviews-Large-Dataset-50k)

- `train.xlsx`: 25,000 labelled reviews
- `test.xlsx`: 25,000 labelled reviews
- Columns: `Reviews` (text) and `Sentiment` (`pos` / `neg`), perfectly balanced

The raw `.xlsx` files are **not** committed to this repository. They are read by
`ml/data_pipeline.py`, which writes the versioned CSVs described below.

## Reproduce

```bash
python -m ml.data_pipeline \
  --train <raw>/train.xlsx \
  --test  <raw>/test.xlsx \
  --out-dir data --limit 8000

dvc add data/train.csv data/train_v2.csv data/test.csv
dvc push
```

`--limit` controls how many training rows are kept (0 = all). We use 8,000 so
that experiments run in seconds; the assignment explicitly allows toy-scale
data.

## Dataset versions

| Version | File | Rows | DVC md5 | Commit |
| --- | --- | --- | --- | --- |
| v1 | `data/train.csv` | 8,000 | `fa9aa5222728f26757b5c328a3242c28` | `4112437` |
| v2 | `data/train_v2.csv` | 6,974 | `e2a4eedda6029ed680ae015c29483e47` | `6b60026` |
| test | `data/test.csv` | 2,000 | `87261e6072ceebe88ec8c418305b2ceb` | `4112437` |

The `.dvc` files in `data/` are the pointers Git tracks; the CSVs themselves
live in DVC cache / Azure Blob storage.

## What changed from v1 to v2

v2 is derived **from the sampled v1** (same `random_state`), so the row-count
difference is exactly what cleaning removed:

1. **HTML tag removal** — IMDB reviews contain `<br />` and similar markup.
2. **URL removal** — links carry no sentiment signal.
3. **Whitespace normalisation** — collapse runs of spaces and newlines.
4. **De-duplication** — identical reviews would bias the model.
5. **Length filtering** — keep reviews of 10–400 words; drop near-empty and
   extremely long ones.

Result: **8,000 → 6,974 rows (−1,026, −12.8%)**, average length 1,317 → 990
characters.

## Linking code to data

Every MLflow run logs both:

- `dataset_md5` — read from the matching `.dvc` pointer
- `git_sha` — the commit that produced the run

so any model can be traced back to the exact dataset version and code revision
that produced it.

## Data hygiene

`data/*.csv` is listed in `.gitignore`; only the `.dvc` pointers are committed.
Raw data is never pushed to GitHub.
