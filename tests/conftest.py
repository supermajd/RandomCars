"""conftest.py: Shared pytest fixtures for the Random Houses test suite."""

__author__ = 'Majd Jamal'

import json

import joblib
import pytest
from backend.main import app
from fastapi.testclient import TestClient
from ml.config import APPROVED_DIR, DATA_PATH
from ml.features import load_data, split


@pytest.fixture
def client():
    """FastAPI test client that triggers startup (lifespan), so the model loads.
    :return client: TestClient bound to the app
    """

    with TestClient(app) as client:
        yield client


@pytest.fixture
def payload() -> dict:
    """A valid /predict request body using the Ames feature aliases.
    :return payload: One house as a feature dict
    """

    return {
        'Gr Liv Area': 1710,
        'Lot Area': 8450,
        'Overall Qual': 7,
        'Overall Cond': 5,
        'Year Built': 2003,
        'Year Remod/Add': 2003,
        'Total Bsmt SF': 856,
        'Garage Cars': 2,
        'Garage Area': 548,
        'Full Bath': 2,
        'TotRms AbvGrd': 8,
        'Fireplaces': 0,
        'Neighborhood': 'CollgCr',
        'House Style': '2Story',
        'Bldg Type': '1Fam',
        'Central Air': 'Y',
        'Exter Qual': 'Gd',
        'Kitchen Qual': 'Gd',
    }


@pytest.fixture(scope='session')
def test_split():
    """Load the deterministic train/test split without training a model."""

    data = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = split(data)

    return X_train, X_test, y_train, y_test


@pytest.fixture(scope='session')
def approved_model():
    """Load the pushed approved model artifact without training."""

    latest_path = APPROVED_DIR / 'latest.json'

    with open(latest_path, encoding='utf-8') as f:
        latest = json.load(f)

    model_id = latest['model_id']

    model_path = APPROVED_DIR / f'{model_id}.joblib'
    metadata_path = APPROVED_DIR / f'{model_id}.metadata.json'

    model = joblib.load(model_path)

    with open(metadata_path, encoding='utf-8') as f:
        metadata = json.load(f)

    return model, metadata, latest
