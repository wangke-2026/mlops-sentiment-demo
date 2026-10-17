import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data(file_path: str):
    df = pd.read_csv(file_path)
    df = df.dropna(subset=["text"])
    return df

def build_vectorizer():
    vec = TfidfVectorizer(stop_words="english", max_features=2000)
    return vec