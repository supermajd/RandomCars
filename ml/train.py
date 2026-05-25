
"""train.py: Trains the Random Houses model, evaluates it against a baseline, and saves the artifact with metadata."""

__author__ = "Majd Jamal"


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

import subprocess

from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.ensemble import RandomForestRegressor

from ml.features import (
    load_data, 
    split, 
    build_preprocessor)

from ml.evaluate import (
    evaluate, 
    baseline_metrics, 
    check_gates
)

from ml.config import (
    DATA_PATH,
    CANDIDATE_DIR,
    RANDOM_STATE,
    FEATURES,
    TEST_SIZE,
    MODEL_CONFIGS,
)

from datetime import datetime, timezone

import joblib

import json 

from pathlib import Path

from ml.args import parse_train_args

def build(preprocessor, model) -> Pipeline:
    """ Builds the full pipeline: preprocessing followed by the regressor.
    :param preprocessor: Unfitted ColumnTransformer
    :param model: Unfitted regressor
    :return pipeline: Unfitted scikit-learn Pipeline
    """

    pipeline = Pipeline(steps = [
        ('preprocessor', preprocessor),
        ('regressor', model),
    ])

    return pipeline

def get_git_sha() -> str:
    """ Returns the short git commit SHA, or 'nogit' if unavailable.
    :return sha: Short commit SHA
    """

    try:
        sha = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr = subprocess.DEVNULL).decode().strip()
    except Exception:
        sha = 'nogit'

    return sha

def save_model(model, metrics, model_name, model_params) -> None:
    """ Saves the model artifact and metadata to the candidates directory.

    The model is saved as a joblib artifact and accompanied by a metadata
    JSON file capturing the information needed for traceability: model name,
    timestamp, git commit, features, seed, parameters, and metrics.

    :param model: Fitted pipeline to save
    :param metrics: Evaluation metrics from evaluate
    """

    CANDIDATE_DIR.mkdir(parents = True, exist_ok = True)

    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H%M%SZ')
    git_sha = get_git_sha()
    model_id = f'{model_name}_{timestamp}_{git_sha}'

    model_path = CANDIDATE_DIR / f'{model_id}.joblib'
    meta_path = CANDIDATE_DIR / f'{model_id}.metadata.json'

    joblib.dump(model, model_path)

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
        'metrics': metrics,
        'status': 'candidate',
    }

    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent = 2)

    Path('artifacts').mkdir(exist_ok = True)
    Path('artifacts/last_candidate.txt').write_text(model_id)

    print(f'Saved candidate model: {model_id}')

    return model_id

def resolve_model_params(model_name, args):
    model_params = MODEL_CONFIGS[model_name].copy()

    cli_overrides = {
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth,
        'random_state': args.random_state,
        'n_jobs': args.n_jobs,
        'learning_rate': args.learning_rate,
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

    if model_name == 'random_forest':
        model = RandomForestRegressor(**model_params)

    elif model_name == 'xgboost':
        model = XGBRegressor(**model_params)
    else:
        raise ValueError(
            f'Unsupported model_name: {model_name}. '
            'Use "random_forest" or "xgboost".'
        )
    pipeline = build(preprocessor, model)
    
    pipeline.fit(X_train, y_train)


    metrics = evaluate(pipeline, X_test, y_test)
    baseline = baseline_metrics(X_train, X_test, y_train, y_test)

    checker = check_gates(metrics, baseline)
    
    if not check_gates(metrics, baseline):
        raise ValueError(
            'Model failed quality gates! It did not beat the baseline on MAE/RMSE, or R2 <= 0.')

    print('\n =-=-=-=- Quality gates passed -=-=-=-= \n')
    save_model(
        pipeline,
        metrics,
        model_name,
        model_params
        )


if __name__ == '__main__':

    main()

