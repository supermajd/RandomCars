"""train.py: Trains the Car Broker 1001 model and evaluates it against a baseline."""

__author__ = 'Majd Jamal'


import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import joblib
import mlflow
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from tensorflow import keras
from xgboost import XGBRegressor

from ml.args import parse_train_args
from ml.config import (
    CANDIDATE_DIR,
    DATA_PATH,
    FEATURES,
    MODEL_CONFIGS,
    RANDOM_STATE,
    TEST_SIZE,
)
from ml.evaluate import baseline_metrics, check_gates, compute_metrics, evaluate
from ml.features import build_preprocessor, load_data, split
from ml.nn import BuildNN
from ml.report import print_report

mlflow.set_tracking_uri('file:./mlruns')  # local default; swap for a server later
mlflow.set_experiment('car-broker-1001')


def build(preprocessor, model) -> Pipeline:
    """Builds the full pipeline: preprocessing followed by the regressor.
    :param preprocessor: Unfitted ColumnTransformer
    :param model: Unfitted regressor
    :return pipeline: Unfitted scikit-learn Pipeline
    """

    pipeline = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('regressor', model),
        ]
    )

    return pipeline


def get_git_sha() -> str:
    """Returns the short git commit SHA, or 'nogit' if unavailable.
    :return sha: Short commit SHA
    """

    try:
        sha = (
            subprocess.check_output(
                ['git', 'rev-parse', '--short', 'HEAD'], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        sha = 'nogit'

    return sha


def save_model(model, preprocessor, metrics, model_name, model_params) -> str:
    """Saves the model artifact and metadata to the candidates directory.

    For sklearn pipelines (random_forest, xgboost), the full pipeline is
    saved as a single joblib artifact. For nn, the Keras model is saved
    in the native .keras format and the fitted preprocessor is saved
    separately as a joblib artifact.

    :param model: Fitted sklearn Pipeline or Keras model
    :param preprocessor: Fitted preprocessor (only used for nn)
    :param metrics: Evaluation metrics from evaluate
    :param model_name: Model identifier ('random_forest', 'xgboost', 'nn')
    :param model_params: Resolved model parameters
    :return model_id: Identifier of the saved candidate
    """

    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC).strftime('%Y-%m-%dT%H%M%SZ')
    git_sha = get_git_sha()
    model_id = f'{model_name}_{timestamp}_{git_sha}'

    meta_path = CANDIDATE_DIR / f'{model_id}.metadata.json'

    artifact_paths = {}

    if model_name == 'nn':
        model_path = CANDIDATE_DIR / f'{model_id}.keras'
        preprocessor_path = CANDIDATE_DIR / f'{model_id}.preprocessor.joblib'

        model.save(model_path)
        joblib.dump(preprocessor, preprocessor_path)

        artifact_paths['model'] = str(model_path)
        artifact_paths['preprocessor'] = str(preprocessor_path)

    else:
        model_path = CANDIDATE_DIR / f'{model_id}.joblib'
        joblib.dump(model, model_path)

        artifact_paths['model'] = str(model_path)

    metadata = {
        'model_id': model_id,
        'model_name': model_name,
        'created_at': timestamp,
        'git_sha': git_sha,
        'data_path': str(DATA_PATH),
        'features': FEATURES,
        'random_state': RANDOM_STATE,
        'test_size': TEST_SIZE,
        'model_params': model_params,
        'artifacts': artifact_paths,
        'metrics': metrics,
        'status': 'candidate',
    }

    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    Path('artifacts').mkdir(exist_ok=True)
    Path('artifacts/last_candidate.txt').write_text(model_id)

    print(f'Saved candidate model: {model_id}')

    return model_id


def resolve_model_params(model_name, args):
    """Resolves model parameters from config and CLI overrides.
    :param model_name: Model identifier
    :param args: Parsed CLI args
    :return model_params: Final parameter dict
    """

    model_params = MODEL_CONFIGS[model_name].copy()

    cli_overrides = {
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth,
        'random_state': args.random_state,
        'n_jobs': args.n_jobs,
        'learning_rate': args.learning_rate,
        'layers': args.layers,
        'dropout': args.dropout,
        'activation': args.activation,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'validation_split': args.validation_split,
        'early_stopping_patience': args.early_stopping_patience,
    }

    for key, value in cli_overrides.items():
        if value is not None:
            model_params[key] = value

    return model_params


def main():

    args = parse_train_args()
    model_name = args.model_name
    model_params = resolve_model_params(model_name, args)

    data = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = split(data)

    preprocessor = build_preprocessor()

    preprocessor.fit(X_train)
    X_train_t = preprocessor.transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    # =-=-=-=-=-=-=-=-=-=-
    # Make model
    # =-=-=-=-=-=-=-=-=-=-

    if model_name == 'random_forest':
        model = RandomForestRegressor(**model_params)
    elif model_name == 'xgboost':
        model = XGBRegressor(**model_params)
    elif model_name == 'nn':
        if hasattr(X_train_t, 'toarray'):
            X_train_t = X_train_t.toarray()
            X_test_t = X_test_t.toarray()

        input_dim = X_train_t.shape[1]

        model = BuildNN(
            input_dim,
            model_params['layers'],
            model_params['learning_rate'],
            model_params['dropout'],
            model_params['activation'],
        )
    else:
        raise ValueError(
            f'Unsupported model_name: {model_name}. Use "random_forest" or "xgboost" or "nn".'
        )
    with mlflow.start_run(run_name=f'{model_name}_{get_git_sha()}') as run:
        mlflow.log_params(model_params)
        mlflow.set_tags(
            {
                'model_name': model_name,
                'git_sha': get_git_sha(),
                'data_path': str(DATA_PATH),
            }
        )

        # =-=-=-=-=-=-=-=-=-=-
        # Train model
        # =-=-=-=-=-=-=-=-=-=-
        if model_name == 'random_forest' or model_name == 'xgboost':
            pipeline = build(preprocessor, model)
            pipeline.fit(X_train, y_train)
            metrics = evaluate(pipeline, X_test, y_test)
        else:
            y_train_log = np.log1p(y_train)

            callbacks = [
                keras.callbacks.EarlyStopping(
                    patience=model_params['early_stopping_patience'],
                    restore_best_weights=True,
                ),
            ]

            model.fit(
                X_train_t,
                y_train_log,
                epochs=model_params['epochs'],
                batch_size=model_params['batch_size'],
                validation_split=model_params['validation_split'],
                callbacks=callbacks,
                verbose=1,
            )

            y_pred_log = model.predict(X_test_t).flatten()
            y_pred = np.expm1(y_pred_log)
            metrics = compute_metrics(y_test, y_pred)

        print_report(metrics, model_name)

        baseline = baseline_metrics(X_train, X_test, y_train, y_test)
        mlflow.log_metrics({f'baseline_{k}': v for k, v in baseline.items()})
        mlflow.log_metrics(metrics)

        gates_passed = check_gates(metrics, baseline)
        mlflow.set_tag('gates_passed', str(gates_passed))

        if not gates_passed:
            raise ValueError(
                'Model failed quality gates! It did not beat the baseline on MAE/RMSE, or R2 <= 0.'
            )

        print('\n =-=-=-=- Quality gates passed -=-=-=-= \n')
        model_id = save_model(
            model if model_name == 'nn' else pipeline,
            preprocessor,
            metrics,
            model_name,
            model_params,
        )
        mlflow.set_tag('model_id', model_id)

        for suffix in ('.keras', '.joblib', '.preprocessor.joblib', '.metadata.json'):
            path = CANDIDATE_DIR / f'{model_id}{suffix}'
            if path.exists():
                mlflow.log_artifact(str(path))

        meta_path = CANDIDATE_DIR / f'{model_id}.metadata.json'
        meta = json.loads(meta_path.read_text())
        meta['mlflow_run_id'] = run.info.run_id
        meta_path.write_text(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
