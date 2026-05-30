"""evaluate.py: Computes model metrics and runs the baseline quality gate."""

__author__ = 'Majd Jamal'

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def check_gates(metrics, baseline) -> bool:
    """Checks the model passes the quality gates.

    The model must beat the baseline on MAE and RMSE, and keep R2 above 0.

    :param metrics: Model metrics from evaluate
    :param baseline: Baseline metrics from baseline_metrics
    :return passed: True if all gates pass
    """

    beats_mae = metrics['mae'] < baseline['mae']
    beats_rmse = metrics['rmse'] < baseline['rmse']
    positive_r2 = metrics['r2'] > 0

    passed = beats_mae and beats_rmse and positive_r2

    return passed


def check_regression(metrics, previous, tolerance: float = 0.10) -> bool:
    """Checks the new model does not regress against the previous approved model.

    The new model must not worsen MAE or RMSE by more than the allowed
    tolerance. If there is no previous model, the check passes.

    :param metrics: New model metrics from evaluate
    :param previous: Previous approved model metrics, or None if none exists
    :param tolerance: Allowed fractional worsening, default 0.10 (10%)
    :return passed: True if the model is within the regression tolerance
    """

    if previous is None:
        return True

    mae_ok = metrics['mae'] <= previous['mae'] * (1 + tolerance)
    rmse_ok = metrics['rmse'] <= previous['rmse'] * (1 + tolerance)

    passed = mae_ok and rmse_ok

    return passed


def baseline_metrics(X_train, X_test, y_train, y_test) -> dict:
    """Fits a DummyRegressor baseline and returns its metrics.
    :param X_train: Training features
    :param X_test: Test features
    :param y_train: Training targets
    :param y_test: Test targets
    :return metrics: Baseline MAE, RMSE, and R2
    """

    dummy = DummyRegressor(strategy='mean')
    dummy.fit(X_train, y_train)

    metrics = evaluate(dummy, X_test, y_test)

    return metrics


def compute_metrics(y_actual, y_pred):
    """Computes regression metrics from true labels and predictions.
    :param y_actual: Ground truth values
    :param y_pred: Predicted values
    :return metrics: MAE, RMSE, and R2 as a dict
    """

    mae = mean_absolute_error(y_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    r2 = r2_score(y_actual, y_pred)

    metrics = {
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
    }

    return metrics


def evaluate(model, X_test, y_test):
    """Predicts on X_test and computes regression metrics.
    :param model: Fitted estimator with .predict
    :param X_test: Test features
    :param y_test: Test targets
    :return metrics: MAE, RMSE, and R2
    """

    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)

    return metrics
