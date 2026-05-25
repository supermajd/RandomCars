
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

REGRESSION_TOLERANCE = 0.05

#-=-=-=-
# Training
#-=-=-=-

RANDOM_STATE = 42
TEST_SIZE = 0.2

MODEL_NAME = "random_forest"  # default only


MODEL_CONFIGS = {
    "random_forest": {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    },

    "xgboost": {
        "n_estimators": 500,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "reg:squarederror",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    },
}

if MODEL_NAME not in MODEL_CONFIGS:
    raise ValueError(
        f"Unsupported MODEL_NAME: {MODEL_NAME}. "
        f"Supported models: {list(MODEL_CONFIGS)}"
    )

MODEL_PARAMS = MODEL_CONFIGS[MODEL_NAME]