"""args.py: a place to put all machine learning training arguments, to make training smoother."""


# ml/args.py

import argparse

from ml.config import MODEL_NAME


def parse_train_args():
    parser = argparse.ArgumentParser(
        description='Train a car price model and save it as candidate if it passes quality gates.'
    )

    parser.add_argument(
        '--model-name',
        default=MODEL_NAME,
        choices=['random_forest', 'xgboost', 'nn'],
        help='Model type to train.',
    )

    # Shared-ish args
    parser.add_argument('--n-estimators', type=int, default=None)
    parser.add_argument('--max-depth', type=int, default=None)
    parser.add_argument('--random-state', type=int, default=None)
    parser.add_argument('--n-jobs', type=int, default=None)

    # XGBoost specific
    parser.add_argument('--learning-rate', type=float, default=None)

    # NN specific

    parser.add_argument(
        '--layers',
        type=int,
        nargs='+',
        default=None,
        help='Hidden layer sizes for nn, e.g. --layers 128 256',
    )
    parser.add_argument('--dropout', type=float, default=None)
    parser.add_argument('--activation', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--validation-split', type=float, default=None)
    parser.add_argument('--early-stopping-patience', type=int, default=None)

    return parser.parse_args()
