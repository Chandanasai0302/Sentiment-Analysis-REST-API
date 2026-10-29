from fastapi.testclient import TestClient
import app as sentiment_app
import numpy as np


class FakeVectorizer:
    def transform(self, texts):
        return texts


class FakeModel:
    def predict(self, features, verbose=0):
        return np.array([[0.8]])


client = TestClient(sentiment_app.app)


def setup_function():
    sentiment_app.model = FakeModel()
    sentiment_app.vectorizer = FakeVectorizer()


def test_predict_returns_expected_schema():
    response = client.post(
        "/predict",
        json={"text": "This product is wonderful"}
    )

    assert response.status_code == 200

    data = response.json()

    assert data["text"] == "This product is wonderful"
    assert data["sentiment"] == "positive"
    assert "positive" in data["probabilities"]
    assert "negative" in data["probabilities"]
    assert data["probabilities"]["positive"] == 0.8
    assert data["probabilities"]["negative"] == 0.2


def test_empty_text_returns_validation_error():
    response = client.post(
        "/predict",
        json={"text": ""}
    )

    assert response.status_code == 422


def test_missing_text_returns_validation_error():
    response = client.post(
        "/predict",
        json={}
    )

    assert response.status_code == 422