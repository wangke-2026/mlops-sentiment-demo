"""Shared pytest fixtures.

The fake model keeps the test suite independent of ml/model.joblib, which is
tracked by DVC and therefore absent on CI runners.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import create_app

POSITIVE_WORDS = ("good", "great", "love", "excellent")


class FakeModel:
    """Deterministic stand-in for the real sklearn Pipeline."""

    def predict(self, texts):
        return [
            1 if any(word in str(text).lower() for word in POSITIVE_WORDS) else 0
            for text in texts
        ]

    def predict_proba(self, texts):
        return [
            [0.2, 0.8] if label == 1 else [0.8, 0.2] for label in self.predict(texts)
        ]


@pytest.fixture
def client():
    return TestClient(create_app(model=FakeModel()))
