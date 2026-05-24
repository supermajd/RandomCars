"""conftest.py: Shared pytest fixtures for the Random Houses test suite."""

__author__ = "Majd Jamal"

import pytest
from fastapi.testclient import TestClient

from backend.main import app


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