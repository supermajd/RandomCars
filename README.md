# Car Broker 1001

A small MLOps project that shows how a machine-learning model is trained, tested, promoted, and served — not just the model, but the workflow around it.

It trains a model on a used-car dataset to predict selling prices, then serves it through a FastAPI backend with automated tests and CI.

The backend is **deployed and public**, so anyone can try it from the terminal with cURL — no setup or API key required.

**Live backend:** https://randomhouses.onrender.com

## The point of the project

The focus is MLOps: the practices that move a model from training to a running service in a repeatable way.

- Train a model and save it as a candidate artifact
- Check model quality before it ships (must beat a naive baseline)
- Promote approved models for serving, with a no-regression gate
- Serve predictions with a FastAPI API
- Log every prediction to SQLite
- Run tests and CI automatically on each push

## Tech stack

Python · scikit-learn · XGBoost · TensorFlow/Keras · pandas · FastAPI · SQLite · pytest · GitHub Actions

## Try the live backend yourself

The service is live at https://randomhouses.onrender.com — just copy the commands below into your terminal.

Health check:

```bash
curl -s https://randomhouses.onrender.com/health | jq
```

```json
{
  "status": "ok",
  "model_version": "nn_2026-05-29T202148Z_8165f6d"
}
```

Predict a price (JSON keys match the dataset column names):

```bash
curl -s -X POST https://randomhouses.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "brand": "Maruti",
    "km_driven": 120000,
    "fuel_type": "Petrol",
    "transmission_type": "Manual",
    "mileage": 19.7,
    "engine": 796,
    "max_power": 46.3,
    "seats": 5
  }' | jq
```

```json
{
  "request_id": "b7f8eee2-68fc-421e-babf-40c1af353c4d",
  "predicted_price": 121500.0,
  "model_version": "nn_2026-05-29T202148Z_8165f6d"
}
```

> The backend is hosted on Render's free tier, so the first request after a while may be slow while it wakes up. If `/predict` returns `503`, the model is still loading — check `/health` first.

## The workflow

```text
train -> evaluate -> save candidate -> promote approved -> serve with API -> validate with CI
```

A model is trained and saved as a candidate. If it passes the quality checks, it is promoted to the approved directory. The API loads the approved model and serves predictions from it.

## API endpoints

```text
GET  /health   # service status and current model version
GET  /model    # model metadata and metrics
POST /predict  # predict a car price
```

## Run it locally

```bash
pip install -r requirements.txt

# train a candidate model (random_forest by default; also xgboost, nn)
python -m ml.train --model-name nn

# promote the latest candidate
python -m ml.persistence --model-id "$(cat artifacts/last_candidate.txt)"

# run the API
uvicorn backend.main:app --reload
```

Then open the interactive docs at http://127.0.0.1:8000/docs.

## Tests

```bash
pytest -v
```

The suite covers the API endpoints, invalid input handling, whether the model beats a baseline, and whether it regresses against the currently approved model.

## CI

`.github/workflows/ci.yml` installs dependencies, lints and format-checks with Ruff, and runs the test suite on every push and pull request, so code and model changes are validated before they merge.

## Project structure

```text
.github/workflows/ci.yml   # CI pipeline

backend/
  main.py                  # FastAPI app: health, model, predict
  schemas.py               # request/response schemas

db/
  db.py                    # SQLite setup and prediction logging

ml/
  config.py                # settings, feature lists, paths, model params
  args.py                  # training CLI arguments
  features.py              # data loading, splitting, preprocessing
  nn.py                    # Keras neural-network model builder
  train.py                 # trains and saves candidate models
  evaluate.py              # metrics and quality checks
  report.py                # training report and metric comparison
  persistence.py           # promotes and loads approved models

models/
  candidates/              # newly trained artifacts
  approved/                # production-approved artifacts

tests/
  api/                     # endpoint tests
  smoke/                   # model quality and regression tests
```
