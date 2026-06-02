---
title: Car Broker 1001
emoji: 🚗
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
---
# Car Broker 1001

A small MLOps project that trains a model on used-car data to predict selling prices, then serves it through a FastAPI backend with automated tests, CI, and Kubernetes deployment.

The point isn't the model — it's the workflow around it: training, quality gates, promotion, serving, observability, and deployment to Kubernetes.

**Live backend:** https://supermajd-randomcars.hf.space

## What this project demonstrates

- Train candidate models and save them with traceable metadata
- Quality gates that block bad models before they ship
- Promote approved models with a no-regression check against the current best
- Serve predictions via FastAPI, with every prediction logged to SQLite
- Run tests and lint in CI on every push
- Deploy to local Kubernetes with `kind`, using Deployment, Service, and PVC

## Tech stack

Python · scikit-learn · XGBoost · TensorFlow/Keras · pandas · FastAPI · SQLite · pytest · GitHub Actions · Docker · Kubernetes (`kind`)

## Try the live backend

Health check:

```bash
curl -s https://supermajd-randomcars.hf.space/health | jq
```

```json
{
  "status": "ok",
  "model_version": "xgboost_2026-05-30T213254Z_8165f6d"
}
```

Predict a price:

```bash
curl -s -X POST https://supermajd-randomcars.hf.space/predict \
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
  "request_id": "5d058398-85e9-4131-b31e-38a3aeff8e20",
  "predicted_price": 125410.265625,
  "model_version": "xgboost_2026-05-30T213254Z_8165f6d"
}
```

> Hosted on Hugging Face Spaces' free tier — the first request may be slow while the service wakes up. If `/predict` returns `503`, the model is still loading. Check `/health` first.

## The workflow

```text
train -> evaluate -> save candidate -> promote approved -> serve with API -> validate with CI
```

A model is trained and saved as a candidate with full metadata. If it passes the quality checks (beats a `DummyRegressor` baseline, R² > 0), it can be promoted to the approved directory. The API loads the approved model and serves predictions from it. Every prediction is logged to SQLite for traceability.

## API endpoints

```text
GET  /health   # service status and current model version
GET  /model    # model metadata and metrics
POST /predict  # predict a car price
```

## Run it locally

```bash
pip install -r requirements.txt

# train a candidate (random_forest by default; also xgboost, nn)
python -m ml.train --model-name xgboost

# promote the latest candidate
python -m ml.persistence --model-id "$(cat artifacts/last_candidate.txt)"

# run the API
uvicorn backend.main:app --reload
```

Then open the interactive docs at http://127.0.0.1:8000/docs.

## Run it on Kubernetes

A local cluster with `kind` runs the full deployment stack — Deployment, Service, and a persistent volume for SQLite.

```bash
# create a local cluster
kind create cluster --name randomcars

# build and load the image into the cluster
docker build -t randomcars:dev .
kind load docker-image randomcars:dev --name randomcars

# apply the manifests
kubectl apply -f deploy/k8s/

# check the pod is running and passing health probes
kubectl get pods

# expose the API locally
kubectl port-forward service/randomcars-api 8080:80
```

Then hit `http://localhost:8080/health` from another terminal.

What's in `deploy/k8s/`:

- `deployment.yaml` — pod lifecycle, image, env, resource limits, readiness and liveness probes
- `service.yaml` — stable internal address that load-balances across pods
- `pvc.yaml` — persistent volume so SQLite survives pod restarts

Tear down when done:

```bash
kind delete cluster --name randomcars
```

The same manifests would run on AKS or any managed Kubernetes; only the Service type (LoadBalancer or Ingress) and the storage class would change.

## Tests

```bash
pytest -v
```

The suite covers API endpoints, invalid input handling, baseline-beating, and regression checks against the currently approved model.

## CI

`.github/workflows/ci.yml` lints with Ruff and runs the test suite on every push and pull request, so code and model changes are validated before merge.

## Project structure

```text
.github/workflows/ci.yml   # CI pipeline

backend/
  main.py                  # FastAPI app: health, model, predict
  schemas.py               # request/response schemas

db/
  db.py                    # SQLite setup and prediction logging

deploy/
  k8s/
    deployment.yaml        # pod lifecycle and probes
    service.yaml           # stable internal address
    pvc.yaml               # persistent volume for SQLite

ml/
  config.py                # settings, features, paths, model params
  args.py                  # training CLI arguments
  features.py              # data loading, splitting, preprocessing
  nn.py                    # Keras neural-network builder
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