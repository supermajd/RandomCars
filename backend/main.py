"""main.py: Random Houses inference API. Loads the approved model and serves predictions."""

__author__ = 'Majd Jamal'

from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException

from backend.schemas import HouseFeatures, ModelInfo, PredictionResponse
from db.db import init_db, log_prediction
from ml.persistence import load_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads the approved model and metadata once before serving."""

    init_db()
    app.state.model, app.state.metadata = load_model()
    yield


app = FastAPI(title='Random Houses', lifespan=lifespan)


@app.get('/health')
def health() -> dict:
    """Liveness/readiness check for smoke tests and probes.
    :return status: Service status and served model version
    """

    model = getattr(app.state, 'model', None)

    if model is None:
        return {'status': 'loading', 'model_version': None}

    return {'status': 'ok', 'model_version': app.state.metadata['model_id']}


@app.get('/model', response_model=ModelInfo)
def model_information():
    """Returns metadata for the currently loaded model.
    :return info: Model version and metrics
    """

    model = getattr(app.state, 'model', None)

    if model is None:
        raise HTTPException(status_code=503, detail='Model not loaded. Service is not ready.')

    metadata = app.state.metadata

    response = ModelInfo(model_version=metadata['model_id'], metrics=metadata['metrics'])

    return response


@app.post('/predict', response_model=PredictionResponse)
def predict(features: HouseFeatures):
    """Predicts a house price from validated input and logs the prediction.
    :param features: Validated house attributes
    :return response: Predicted price, request id, and model version
    """

    model = getattr(app.state, 'model', None)

    if model is None:
        raise HTTPException(status_code=503, detail='Model not loaded. Service is not ready.')

    # -=-=-=-
    # Build model input
    # -=-=-=-
    data = features.model_dump(by_alias=True)
    
    row = pd.DataFrame([data])

    # -=-=-=-
    # Predict
    # -=-=-=-

    pred = model.predict(row)
    predicted_price = float(pred[0])

    # -=-=-=-
    # Log and respond
    # -=-=-=-

    model_version = app.state.metadata['model_id']

    request_id = log_prediction(data, predicted_price, model_version)

    response = PredictionResponse(
        request_id=request_id, predicted_price=predicted_price, model_version=model_version
    )

    return response
