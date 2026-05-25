
""" args.py: a place to put all machine learning training arguments, to make training smoother."""


# ml/args.py

import argparse

from ml.config import MODEL_NAME


def parse_train_args():
    parser = argparse.ArgumentParser(
        description="Train a house price model and save it as candidate if it passes quality gates."
    )

    parser.add_argument(
        "--model-name",
        default=MODEL_NAME,
        choices=["random_forest", "xgboost"],
        help="Model type to train.",
    )

    # Shared-ish args
    parser.add_argument("--n-estimators", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=None)

    # XGBoost specific
    parser.add_argument("--learning-rate", type=float, default=None)

    return parser.parse_args()