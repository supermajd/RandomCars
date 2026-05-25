"""test_model_quality.py: Quality gate — model must outperform a naive baseline."""

__author__ = 'Majd Jamal'

import json
import os

import numpy as np
import pytest
from ml.config import REGRESSION_TOLERANCE
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

# Model must cut the baseline's error by at least this fraction.
# Scale-free: holds on any dataset, unlike an absolute MAE ceiling.
MIN_IMPROVEMENT = 0.40  # at least 40% lower MAE than predicting the mean


def test_model_beats_baseline(approved_model, test_split):
    """The pushed approved model must beat a mean-predicting baseline."""

    pipeline, metadata, latest = approved_model
    X_train, X_test, y_train, y_test = test_split

    dummy = DummyRegressor(strategy='mean')
    dummy.fit(X_train, y_train)

    pred = pipeline.predict(X_test)
    base_pred = dummy.predict(X_test)

    model_mae = mean_absolute_error(y_test, pred)
    base_mae = mean_absolute_error(y_test, base_pred)

    assert np.all(np.isfinite(pred)), 'Predictions contain NaN or inf'

    improvement = 1 - (model_mae / base_mae)

    assert improvement >= MIN_IMPROVEMENT, (
        f'Model MAE {model_mae:.0f} only {improvement:.0%} better than '
        f'baseline {base_mae:.0f}; need >= {MIN_IMPROVEMENT:.0%}'
    )

    assert r2_score(y_test, pred) > 0, 'R2 not positive — worse than the mean'


def test_model_does_not_regress(approved_model, test_split):
    """The pushed approved model must not worsen MAE or RMSE by more than
    REGRESSION_TOLERANCE versus the previous approved model.
    """

    pipeline, metadata, latest = approved_model
    X_train, X_test, y_train, y_test = test_split

    pred = pipeline.predict(X_test)

    assert np.all(np.isfinite(pred)), 'Predictions contain NaN or inf'

    mae = mean_absolute_error(y_test, pred)
    rmse = root_mean_squared_error(y_test, pred)

    previous_best_path = os.environ.get('PREVIOUS_BEST_PATH')

    if previous_best_path is None:
        pytest.fail('PREVIOUS_BEST_PATH not set; cannot compare against previous best')

    with open(previous_best_path, encoding='utf-8') as f:
        front = json.load(f)

    if not front:
        return

    front_metrics = front['metrics']

    assert mae <= front_metrics['mae'] * (1 + REGRESSION_TOLERANCE), (
        f'MAE {mae:.0f} regressed beyond {REGRESSION_TOLERANCE:.0%} '
        f'vs approved {front_metrics["mae"]:.0f}'
    )

    assert rmse <= front_metrics['rmse'] * (1 + REGRESSION_TOLERANCE), (
        f'RMSE {rmse:.0f} regressed beyond {REGRESSION_TOLERANCE:.0%} '
        f'vs approved {front_metrics["rmse"]:.0f}'
    )


def test_model_metadata_matches_actual_metrics(approved_model, test_split):
    """The pushed model metadata must match the actual metrics of the pushed model."""

    pipeline, metadata, latest = approved_model
    X_train, X_test, y_train, y_test = test_split

    pred = pipeline.predict(X_test)

    actual_mae = mean_absolute_error(y_test, pred)
    actual_rmse = root_mean_squared_error(y_test, pred)
    actual_r2 = r2_score(y_test, pred)

    recorded = metadata['metrics']

    assert actual_mae == pytest.approx(recorded['mae'])
    assert actual_rmse == pytest.approx(recorded['rmse'])
    assert actual_r2 == pytest.approx(recorded['r2'])
