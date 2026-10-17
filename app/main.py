from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI(title="Sentiment Model API")

# 加载训练好的模型
model = joblib.load("ml/model.joblib")

@app.get("/")
def root():
    return {"msg": "MLOps sentiment model service"}

@app.post("/predict")
def predict(text: str):
    df = pd.DataFrame([{"text": text}])
    pred = model.predict(df["text"])
    return {"text": text, "prediction": int(pred[0])}