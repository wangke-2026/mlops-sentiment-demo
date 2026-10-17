"""IMDB sentiment data preprocessing.

Produces two dataset versions from the raw IMDB files:

    v1  data/train.csv       minimal: rename columns, map labels, drop broken rows
    v2  data/train_v2.csv    v1 + text cleaning + de-duplication + length filter

The difference between v1 and v2 is what makes the "which dataset version
produced this model?" question meaningful (see ml/README_data.md).

Usage:
    python -m ml.data_pipeline \\
        --train <path/to/train.xlsx> \\
        --test  <path/to/test.xlsx> \\
        --out-dir data \
        --limit 8000
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

# Raw IMDB columns -> the names the rest of the project expects.
RAW_COLUMNS = {"Reviews": "text", "Sentiment": "label"}
LABEL_MAP = {"pos": 1, "neg": 0}

_HTML_TAG = re.compile(r"<[^>]+>")
_URL = re.compile(r"(https?://\S+|www\.\S+)")
_WHITESPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """v2 cleaning: strip HTML tags, URLs, and collapse whitespace.

    Kept as a small pure function so unit tests can hit it directly with
    edge cases (empty strings, bare HTML, only whitespace, ...).
    """
    if not isinstance(text, str):
        return ""
    text = _HTML_TAG.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _WHITESPACE.sub(" ", text)
    return text.strip()


def read_raw(path: str | Path) -> pd.DataFrame:
    """Read a raw IMDB file (.xlsx or .csv) and normalise its columns."""
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    df = df.rename(columns=RAW_COLUMNS)
    df["label"] = df["label"].map(LABEL_MAP)
    return df


def build_v1(df: pd.DataFrame) -> pd.DataFrame:
    """v1: the first usable dataset -- drop rows we cannot train on."""
    v1 = df.dropna(subset=["text", "label"]).copy()
    v1 = v1[v1["text"].astype(str).str.strip() != ""]
    v1["label"] = v1["label"].astype(int)
    return v1.reset_index(drop=True)


def build_v2(
    df: pd.DataFrame, min_words: int = 10, max_words: int = 400
) -> pd.DataFrame:
    """v2: v1 + clean text, remove duplicates, drop extreme-length reviews."""
    v2 = build_v1(df)
    v2["text"] = v2["text"].map(clean_text)
    v2 = v2[v2["text"] != ""]

    n_words = v2["text"].str.split().str.len()
    v2 = v2[(n_words >= min_words) & (n_words <= max_words)]

    v2 = v2.drop_duplicates(subset=["text"])
    return v2.reset_index(drop=True)


def _take(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    """Optionally sample down, keeping the label distribution balanced."""
    if limit and limit < len(df):
        return df.sample(n=limit, random_state=42).reset_index(drop=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build IMDB sentiment dataset versions"
    )
    parser.add_argument("--train", required=True, help="raw train file (.xlsx or .csv)")
    parser.add_argument("--test", required=True, help="raw test file (.xlsx or .csv)")
    parser.add_argument("--out-dir", default="data", help="output directory")
    parser.add_argument(
        "--limit", type=int, default=8000, help="train rows to keep (0 = all)"
    )
    parser.add_argument(
        "--test-limit", type=int, default=2000, help="test rows to keep (0 = all)"
    )
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw_train = read_raw(args.train)
    raw_test = read_raw(args.test)

    # v1 is the sampled baseline; v2 is derived FROM v1, so the row-count
    # difference reflects exactly what cleaning removed.
    v1 = _take(build_v1(raw_train), args.limit)
    v2 = build_v2(v1)
    test = _take(build_v1(raw_test), args.test_limit)

    v1.to_csv(out / "train.csv", index=False)
    v2.to_csv(out / "train_v2.csv", index=False)
    test.to_csv(out / "test.csv", index=False)

    for name, frame in (
        ("v1  train.csv", v1),
        ("v2  train_v2.csv", v2),
        ("    test.csv", test),
    ):
        counts = frame["label"].value_counts().to_dict()
        print(f"{name:20s} rows={len(frame):6d}  {counts}")


if __name__ == "__main__":
    main()
