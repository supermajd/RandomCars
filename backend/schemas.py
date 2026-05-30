"""schemas.py: Request and response schemas for the Car Broker 1001 API."""

__author__ = 'Majd Jamal'

from pydantic import BaseModel


class CarFeatures(BaseModel):
    """Validated input features for a single car.

    Field names match the exact column names the model was trained on
    (see ml/config.py). Numeric features are km_driven, mileage, engine,
    max_power, and seats. Categorical features are brand, fuel_type, and
    transmission_type.
    """

    # Numeric features
    km_driven: int
    mileage: float
    engine: float
    max_power: float
    seats: int

    # Categorical features
    brand: str
    fuel_type: str
    transmission_type: str


class PredictionResponse(BaseModel):
    request_id: str
    predicted_price: float
    model_version: str


class ModelInfo(BaseModel):
    model_version: str
    metrics: dict
