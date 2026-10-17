"""Train the IMDB sentiment model and log the run to MLflow.

Every run records:
    - the git commit SHA            (which code)
    - the dataset md5 from DVC      (which data)
    - the hyperparameters           (what you tuned)
    - accuracy and F1               (how it did)
    - the trained model             (the artifact)

The model is saved as a single sklearn Pipeline, so the API can call
model.predict(["some review"]) on raw text.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline


def git_sha() -> str:
    """The current commit, so a run can be traced back to its code."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # noqa: BLE001 - git may not exist in some environments
        return "unknown"


def dvc_hash(data_file: str | Path) -> str:
    """The md5 DVC recorded for this dataset, so a run traces back to its data."""
    dvc_file = Path(f"{data_file}.dvc")
    if not dvc_file.exists():
        return "unknown"
    with dvc_file.open(encoding="utf-8") as handle:
        spec = yaml.safe_load(handle)
    return spec["outs"][0]["md5"]


def build_model(max_features: int, c_value: float) -> Pipeline:
    """One Pipeline: raw text in, label out. The API depends on this shape."""
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(stop_words="english", max_features=max_features)),
            ("clf", LogisticRegression(max_iter=1000, C=c_value)),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and log the sentiment model")
    parser.add_argument("--train", default="data/train.csv", help="training CSV")
    parser.add_argument("--test", default="data/test.csv", help="evaluation CSV")
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--run-name", default="baseline")
    parser.add_argument("--experiment", default="sentiment")
    parser.add_argument("--model-out", default="ml/model.joblib")
    args = parser.parse_args()

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)

    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name=args.run_name):
        # Provenance: which code + which data produced this run.
        mlflow.log_param("git_sha", git_sha())
        mlflow.log_param("dataset_file", args.train)
        mlflow.log_param("dataset_md5", dvc_hash(args.train))

        # Hyperparameters.
        mlflow.log_param("max_features", args.max_features)
        mlflow.log_param("C", args.C)

        model = build_model(args.max_features, args.C)
        model.fit(train["text"], train["label"])

        predictions = model.predict(test["text"])
        accuracy = accuracy_score(test["label"], predictions)
        f1 = f1_score(test["label"], predictions)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1", f1)

        # Artifacts: the MLflow model and the raw joblib the API loads.
        mlflow.sklearn.log_model(model, name="model")
        Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, args.model_out)
        mlflow.log_artifact(args.model_out)

        print(f"run      = {args.run_name}")
        print(f"data     = {args.train} (md5 {dvc_hash(args.train)[:8]})")
        print(f"accuracy = {accuracy:.4f}")
        print(f"f1       = {f1:.4f}")
        print(f"model    -> {args.model_out}")


if __name__ == "__main__":
    main()
