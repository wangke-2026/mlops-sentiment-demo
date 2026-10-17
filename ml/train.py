import joblib
from sklearn.linear_model import LogisticRegression
from ml.registry.data_pipeline import load_data, build_vectorizer

def train():
    df = load_data("data/train.csv")
    X_raw = df["text"]
    y = df["label"]

    vec = build_vectorizer()
    X = vec.fit_transform(X_raw)

    model = LogisticRegression()
    model.fit(X, y)

    joblib.dump(model, "ml/model.joblib")
    joblib.dump(vec, "ml/vectorizer.joblib")
    print("Model saved to ml/model.joblib")

if __name__ == "__main__":
    train()