"""conftest.py: Shared pytest fixtures for the Car Broker 1001 test suite."""

__author__ = 'Majd Jamal'

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from ml.config import DATA_PATH
from ml.features import load_data, split
from ml.persistence import load_model


@pytest.fixture
def client():
    """FastAPI test client that triggers startup (lifespan), so the model loads.
    :return client: TestClient bound to the app
    """

    with TestClient(app) as client:
        yield client


@pytest.fixture
def payload() -> dict:
    """A valid /predict request body matching the car feature schema.
    :return payload: One car as a feature dict
    """

    return {
        'brand': 'Maruti',
        'km_driven': 120000,
        'fuel_type': 'Petrol',
        'transmission_type': 'Manual',
        'mileage': 19.7,
        'engine': 796,
        'max_power': 46.3,
        'seats': 5,
    }


@pytest.fixture(scope='session')
def test_split():
    """Load the deterministic train/test split without training a model."""

    data = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = split(data)

    return X_train, X_test, y_train, y_test


@pytest.fixture(scope='session')
def approved_model():
    """Load the pushed approved model artifact without training.

    Delegates to ml.persistence.load_model so it works for both sklearn
    pipelines (single .joblib) and Keras models (.keras + separate
    preprocessor joblib), keeping the test fixture loader-agnostic.
    """

    model, metadata = load_model()

    if model is None:
        pytest.fail('No approved model found. Train and promote one before running tests.')

    latest = {
        'model_id': metadata['model_id'],
        'model_name': metadata.get('model_name'),
        'metrics': metadata['metrics'],
    }

    return model, metadata, latest
