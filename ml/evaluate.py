
"""evaluate.py: Computes model metrics and runs the baseline quality gate."""

__author__ = "Majd Jamal"

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate(model, X_test, y_test):

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    metrics = {
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
    }

    return metrics

def check_gates(metrics, baseline) -> bool:
    """ Checks the model passes the quality gates.

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

def baseline_metrics(X_train, X_test, y_train, y_test) -> dict:
    """ Fits a DummyRegressor baseline and returns its metrics.
    :param X_train: Training features
    :param X_test: Test features
    :param y_train: Training targets
    :param y_test: Test targets
    :return metrics: Baseline MAE, RMSE, and R2
    """

    dummy = DummyRegressor(strategy = 'mean')
    dummy.fit(X_train, y_train)

    metrics = evaluate(dummy, X_test, y_test)

    return metrics

