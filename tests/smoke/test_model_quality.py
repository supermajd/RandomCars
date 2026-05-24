"""test_model_quality.py: Quality gate — model must outperform a naive baseline."""

__author__ = "Majd Jamal"

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from ml.config import REGRESSION_TOLERANCE
from ml.persistence import load_best


# Model must cut the baseline's error by at least this fraction.
# Scale-free: holds on any dataset, unlike an absolute MAE ceiling.
MIN_IMPROVEMENT = 0.40   # at least 40% lower MAE than predicting the mean


def test_model_beats_baseline(trained_model):
    """ A trained model must beat a mean-predicting baseline by a clear margin.
    :param trained_model: (pipeline, X_train, X_test, y_train, y_test)
    """

    pipeline, X_train, X_test, y_train, y_test = trained_model

    #-=-=-=-
    # Baseline: predict the training mean
    #-=-=-=-

    dummy = DummyRegressor(strategy = 'mean')
    dummy.fit(X_train, y_train)

    pred = pipeline.predict(X_test)
    base_pred = dummy.predict(X_test)

    model_mae = mean_absolute_error(y_test, pred)
    base_mae = mean_absolute_error(y_test, base_pred)

    #-=-=-=-
    # Hygiene + relative quality gate
    #-=-=-=-

    assert np.all(np.isfinite(pred)), 'Predictions contain NaN or inf'

    improvement = 1 - (model_mae / base_mae)

    assert improvement >= MIN_IMPROVEMENT, (
        f'Model MAE {model_mae:.0f} only {improvement:.0%} better than '
        f'baseline {base_mae:.0f}; need >= {MIN_IMPROVEMENT:.0%}')

    assert r2_score(y_test, pred) > 0, 'R2 not positive — worse than the mean'

def test_model_does_not_regress(trained_model):
    """ A freshly trained model must not worsen MAE or RMSE by more than
    REGRESSION_TOLERANCE versus the currently approved model.
    :param trained_model: (pipeline, X_test, y_test)
    """

    pipeline, X_train, X_test, y_train, y_test = trained_model 

    pred = pipeline.predict(X_test)

    assert np.all(np.isfinite(pred)), 'Predictions contain NaN or inf'

    mae = mean_absolute_error(y_test, pred)
    rmse = root_mean_squared_error(y_test, pred)

    front = load_best()

    #-=-=-=-
    # Cold start: nothing approved yet, nothing to regress against
    #-=-=-=-

    if front is None:
        return
        
    assert mae <= front["mae"] * (1 + REGRESSION_TOLERANCE), (
        f'MAE {mae:.0f} regressed beyond {REGRESSION_TOLERANCE:.0%} '
        f'vs approved {front["mae"]:.0f}')

    assert rmse <= front["rmse"] * (1 + REGRESSION_TOLERANCE), (
        f'RMSE {rmse:.0f} regressed beyond {REGRESSION_TOLERANCE:.0%} '
        f'vs approved {front["rmse"]:.0f}')