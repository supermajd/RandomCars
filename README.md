# RandomHouses

RandomHouses is a small MLOps project that demonstrates how to train, test, promote, and serve a machine-learning model with CI/CD practices.

The project uses the Ames Housing dataset to train a `RandomForestRegressor` for house price prediction. The focus is not only the model itself, but the workflow around it: automated tests, model quality checks, model promotion, and GitHub Actions CI.

## What this project demonstrates

- Training a machine-learning model with scikit-learn
- Saving candidate model artifacts
- Promoting approved models for serving
- Serving predictions with FastAPI
- Logging predictions to SQLite
- Testing both API behavior and model quality
- Running automated CI checks with GitHub Actions on every push or pull request

## Tech stack

- Python
- scikit-learn
- pandas
- FastAPI
- SQLite
- pytest
- GitHub Actions

## Repository structure

```text
.github/workflows/ci.yml     # CI pipeline for automated testing

backend/
  main.py                    # FastAPI app with health, model, and predict endpoints
  schemas.py                 # API request and response schemas

db/
  db.py                      # SQLite setup and prediction logging

ml/
  config.py                  # Project settings, feature lists, paths, and model params
  features.py                # Data loading, splitting, and preprocessing
  evaluate.py                # Metrics and model quality checks
  train.py                   # Trains and saves candidate models
  persistence.py             # Promotes and loads approved models

models/
  candidates/                # Newly trained model artifacts
  approved/                  # Production-approved model artifacts

tests/
  api/                       # FastAPI endpoint tests
  smoke/                     # Model quality and regression tests
  conftest.py                # Shared pytest fixtures
```

## MLOps workflow

The intended workflow is:

```text
train model -> evaluate model -> save candidate -> promote approved model -> serve with API -> test with CI
```

A candidate model is trained and saved first. If it passes the quality gates, it can be promoted to the approved model directory. The FastAPI backend then loads the approved model for predictions.

## Running locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Train a model:

```bash
python -m ml.train
```

Promote the latest candidate:

```bash
python -m ml.persistence --model-id "$(cat artifacts/last_candidate.txt)"
```

Run the API:

```bash
uvicorn backend.main:app --reload
```

Open the API docs:

```text
http://127.0.0.1:8000/docs
```

## Running tests

```bash
pytest -v
```

The test suite checks:

- the API health endpoint
- the prediction endpoint
- invalid input handling
- whether the model beats a baseline
- whether the model regresses against the approved model

## CI/CD

The repository includes a GitHub Actions workflow at:

```text
.github/workflows/ci.yml
```

On every push or pull request, GitHub Actions installs the project dependencies and runs the test suite.

This demonstrates a basic MLOps CI pipeline where code and model changes are automatically validated before being merged.

## API endpoints

```text
GET  /health   # Check service health
GET  /model    # Return current model metadata
POST /predict  # Predict house price
```

## Purpose

This repository is built as a portfolio project to show practical MLOps skills: model training, artifact promotion, API serving, automated testing, and CI/CD with GitHub Actions.
