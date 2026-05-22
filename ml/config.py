
"""config.py: Central configuration for the Random Houses project."""

__author__ = "Majd Jamal"

from pathlib import Path


#-=-=-=-
# Paths
#-=-=-=-

DATA_PATH = Path('data/raw/AmesHousing.csv')      # Immutable training input
CANDIDATE_DIR = Path('models/candidates')          # New models land here first
APPROVED_DIR = Path('models/approved')             # Promoted models only
METRICS_DIR = Path('artifacts/metrics')            # Saved evaluation metrics


#-=-=-=-
# Target and features
#-=-=-=-

TARGET = 'SalePrice'

# Numeric features used by the model
NUMERIC_FEATURES = [
    'Gr Liv Area',
    'Lot Area',
    'Overall Qual',
    'Overall Cond',
    'Year Built',
    'Year Remod/Add',
    'Total Bsmt SF',
    'Garage Cars',
    'Garage Area',
    'Full Bath',
    'TotRms AbvGrd',
    'Fireplaces',
]

# Categorical features used by the model
CATEGORICAL_FEATURES = [
    'Neighborhood',
    'House Style',
    'Bldg Type',
    'Central Air',
    'Exter Qual',
    'Kitchen Qual',
]

# Full feature list consumed by training and inference
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


#-=-=-=-
# Training
#-=-=-=-

MODEL_NAME = 'random_forest'     # Logged in metadata; swap to 'knn' if the model changes

RANDOM_STATE = 42                # Fixed seed for split and model determinism
TEST_SIZE = 0.2                  # Holdout fraction for evaluation

# RandomForestRegressor parameters
MODEL_PARAMS = {
    'n_estimators': 300,
    'max_depth': None,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'verbose': 1,
}

