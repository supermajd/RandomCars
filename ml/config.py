"""config.py: Central configuration for the Car Broker 1001 project."""

__author__ = 'Majd Jamal'

from pathlib import Path

# -=-=-=-
# Paths
# -=-=-=-

DATA_PATH = Path('data/raw/Cars.csv')  # Immutable training input
CANDIDATE_DIR = Path('models/candidates')  # New models land here first
APPROVED_DIR = Path('models/approved')  # Promoted models only
METRICS_DIR = Path('artifacts/metrics')  # Saved evaluation metrics


# -=-=-=-
# Target and features
# -=-=-=-

TARGET = 'selling_price'

# Numeric features used by the model
NUMERIC_FEATURES = [
    'km_driven',
    'mileage',
    'engine',
    'max_power',
    'seats',
]

# Categorical features used by the model
CATEGORICAL_FEATURES = [
    'brand',
    'fuel_type',
    'transmission_type',
]

# Full feature list consumed by training and inference
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

REGRESSION_TOLERANCE = 0.05

# -=-=-=-
# Training
# -=-=-=-

RANDOM_STATE = 42
TEST_SIZE = 0.2

MODEL_NAME = 'random_forest'  # default only


MODEL_CONFIGS = {
    'random_forest': {
        'n_estimators': 300,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
    },
    'xgboost': {
        'n_estimators': 500,
        'max_depth': 4,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'reg:squarederror',
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
    },
    'nn': {
        'layers': [128, 256],
        'learning_rate': 1e-3,
        'dropout': 0.2,
        'activation': 'relu',
        'epochs': 200,
        'batch_size': 32,
        'validation_split': 0.2,
        'early_stopping_patience': 10,
    },
}

if MODEL_NAME not in MODEL_CONFIGS:
    raise ValueError(
        f'Unsupported MODEL_NAME: {MODEL_NAME}. Supported models: {list(MODEL_CONFIGS)}'
    )

MODEL_PARAMS = MODEL_CONFIGS[MODEL_NAME]
