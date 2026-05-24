"""conftest.py: Shared pytest fixtures for the Random Houses test suite."""

__author__ = "Majd Jamal"

import pytest
from fastapi.testclient import TestClient

from backend.main import app


from sklearn.ensemble import RandomForestRegressor

from ml.config import DATA_PATH, RANDOM_STATE, MODEL_PARAMS
from ml.features import load_data, split, build_preprocessor
from ml.train import build


@pytest.fixture
def client():
    """ FastAPI test client that triggers startup (lifespan), so the model loads.
    :return client: TestClient bound to the app
    """

    with TestClient(app) as client:
        yield client


@pytest.fixture
def payload() -> dict:
    """ A valid /predict request body using the Ames feature aliases.
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


@pytest.fixture(scope="session")
def trained_model():
    data = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = split(data)

    preprocessor = build_preprocessor()
    model = RandomForestRegressor(**MODEL_PARAMS)

    pipeline = build(preprocessor, model)
    pipeline.fit(X_train, y_train)

    return pipeline, X_train, X_test, y_train, y_test