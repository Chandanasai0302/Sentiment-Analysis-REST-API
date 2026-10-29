# Sentiment Analysis REST API

## 1. Project Overview

The Sentiment Analysis REST API is a machine learning application that predicts whether a text review expresses positive or negative sentiment.

The project uses a trained TensorFlow/Keras neural network model and a TF-IDF vectorizer. FastAPI provides the REST API, and Docker packages the application and its dependencies into a container.

**Important limitation:** The current model was trained using a small demonstration dataset of 30 manually created reviews. Its predictions may be inaccurate and should not be used for important real-world decisions.

## 2. Technologies Used

- Python 3.13
- FastAPI
- Uvicorn
- TensorFlow/Keras
- Scikit-learn
- TF-IDF Vectorizer
- NumPy
- Joblib
- Pytest
- Docker

## 3. Project Structure

```text
Sentiment_API/
├── app.py
├── model.keras
├── vectorizer.pkl
├── requirements.txt
├── Dockerfile
├── test_app.py
└── README.md
```

### File Description

- `app.py`: Implements the FastAPI application and prediction endpoint.
- `model.keras`: Contains the trained neural network model.
- `vectorizer.pkl`: Contains the fitted TF-IDF vectorizer.
- `requirements.txt`: Lists the Python dependencies and pinned versions.
- `Dockerfile`: Defines how to build the Docker image.
- `test_app.py`: Contains automated API tests.
- `README.md`: Documents the project and instructions.

## 4. System Architecture

The application follows this prediction workflow:

1. A user submits a text review to the API.
2. FastAPI validates the input.
3. The TF-IDF vectorizer converts the text into numerical features.
4. The trained neural network predicts a positive sentiment probability.
5. The API calculates the negative sentiment probability.
6. The API returns the predicted sentiment and both probabilities as JSON.

**Architecture flow:**

User → FastAPI `/predict` → Input Validation → TF-IDF Vectorizer → Neural Network Model → JSON Response

The model and vectorizer are loaded when the application starts.

## 5. Installation and Setup

### Prerequisites

Install the following:

- Python 3.13
- pip
- Docker Desktop, if you want to run the Docker container

### Install Dependencies

Open a terminal in the project directory and run:

```bash
python -m pip install -r requirements.txt
```

## 6. Run the API Locally

Start the API with:

```bash
python -m uvicorn app:app --reload
```

Open the following address in your browser:

`http://127.0.0.1:8000`

Interactive API documentation is available at:

`http://127.0.0.1:8000/docs`

Health-check endpoint:

`http://127.0.0.1:8000/health`

## 7. API Endpoints

### GET /

Checks whether the API application is running.

Example response:

```json
{
  "message": "Sentiment Analysis API is running",
  "documentation": "/docs"
}
```

### GET /health

Reports whether the model and vectorizer are loaded.

Example response when both are loaded:

```json
{
  "status": "ready",
  "model_loaded": true,
  "vectorizer_loaded": true
}
```

### POST /predict

Predicts whether the submitted review has positive or negative sentiment.

**Request body:**

```json
{
  "text": "This product is amazing and I really love it"
}
```

**Example response:**

```json
{
  "text": "This product is amazing and I really love it",
  "sentiment": "positive",
  "probabilities": {
    "positive": 0.8,
    "negative": 0.2
  }
}
```

The probabilities above illustrate the response format; actual values depend on the loaded model.

**Response fields:**

- `text`: The submitted review after trimming surrounding whitespace.
- `sentiment`: The predicted class, either `positive` or `negative`.
- `probabilities.positive`: The model's positive sentiment score.
- `probabilities.negative`: The complementary negative sentiment score.

**Common HTTP status codes:**

- `200 OK`: Prediction completed successfully.
- `422 Unprocessable Entity`: The request is missing the text field or contains invalid input.
- `503 Service Unavailable`: The model or vectorizer is not loaded.

## 8. Docker Deployment

Docker packages the application, model, vectorizer, and Python dependencies into a container image.

### Build the Docker Image

Run this command from the directory containing the Dockerfile:

```bash
docker build -t sentiment-api .
```

### Run the Container

```bash
docker run --rm -p 8000:8000 sentiment-api
```

### Access the API

Open:

`http://127.0.0.1:8000/docs`

Use the interactive Swagger documentation to submit review text and inspect the JSON response.

To stop the running container, press `Ctrl + C` in the terminal where it is running.

## 9. Automated Testing

The project uses Pytest and FastAPI's TestClient to check API behavior.

Run the tests with:

```bash
python -m pytest -v
```

The current test suite checks:

1. Successful prediction and the expected response schema.
2. Rejection of empty text.
3. Rejection of requests that omit the text field.

**Latest local test result:** Three tests passed.

The tests use a mock model and vectorizer for predictable outputs. They validate API behavior but do not measure the real model's predictive accuracy.

## 10. Limitations and Future Improvements

### Current Limitations

- The demonstration dataset contains only 30 manually created reviews.
- The model's predictive accuracy and generalization have not been established.
- Passing API tests does not guarantee accurate sentiment predictions.
- The current implementation does not include authentication, rate limiting, production monitoring, or a formal model-performance evaluation.

### Future Improvements

- Train and evaluate the model on a larger, representative, labeled dataset.
- Measure precision, recall, F1-score, and a confusion matrix.
- Add input logging and monitoring while protecting user privacy.
- Introduce authentication and rate limiting.
- Add continuous integration to run tests automatically.
- Deploy the container to a suitable hosting platform.

## 11. Conclusion

This project demonstrates how to package a trained sentiment analysis model as a REST API using FastAPI, validate API behavior with automated tests, and package the application using Docker.

It provides a foundation for further work on model evaluation, reliability, security, and deployment.