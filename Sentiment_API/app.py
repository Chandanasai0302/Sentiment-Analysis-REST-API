import joblib
import numpy as np
import tensorflow as tf
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.keras"
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"

app = FastAPI(
    title="Sentiment Analysis API",
    description="Predicts whether a review is positive or negative.",
    version="1.0.0"
)

model = None
vectorizer = None


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)


@app.on_event("startup")
def load_model():
    global model, vectorizer

    if MODEL_PATH.exists() and VECTORIZER_PATH.exists():
        model = tf.keras.models.load_model(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)


@app.get("/")
def home():
    return {
        "message": "Sentiment Analysis API is running",
        "documentation": "/docs"
    }


@app.get("/health")
def health():
    ready = model is not None and vectorizer is not None

    return {
        "status": "ready" if ready else "model_not_loaded",
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    if model is None or vectorizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model files are not loaded."
        )

    text = request.text.strip()

    if not text:
        raise HTTPException(
            status_code=422,
            detail="Please provide a non-empty review."
        )

    features = vectorizer.transform([text])
    prediction = model.predict(features, verbose=0)

    positive_probability = float(
        np.asarray(prediction).reshape(-1)[0]
    )

    positive_probability = max(
        0.0, min(1.0, positive_probability)
    )
    negative_probability = 1.0 - positive_probability

    sentiment = (
        "positive" if positive_probability >= 0.5 else "negative"
    )

    return {
        "text": text,
        "sentiment": sentiment,
        "probabilities": {
            "positive": round(positive_probability, 4),
            "negative": round(negative_probability, 4)
        }
    }