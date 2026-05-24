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