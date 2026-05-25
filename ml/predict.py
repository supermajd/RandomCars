"""predict.py: Loads an approved model and runs inference on house features."""

__author__ = 'Majd Jamal'

import joblib
import pandas as pd


def load_model(model_path):
    """Loads a trained, approved model artifact.
    :param model_path: Path to the saved .joblib model
    :return model: Fitted scikit-learn Pipeline
    """

    model = joblib.load(model_path)

    return model


def predict(model, features: dict) -> float:
    """Predicts the house price for a single feature record.
    :param model: Fitted pipeline
    :param features: Dictionary of feature name to value
    :return price: Predicted house price
    """

    X = pd.DataFrame([features])
    pred = model.predict(X)
    price = float(pred[0])

    return price
