"""persistence.py: Promotes candidate models to approved and maintains the latest pointer."""

__author__ = 'Majd Jamal'

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import joblib
import mlflow
import numpy as np

from ml.config import APPROVED_DIR, CANDIDATE_DIR, REGRESSION_TOLERANCE

LATEST_PATH = APPROVED_DIR / 'latest.json'


def load_best():
    """Reads the metrics of the currently approved model.
    :return metrics: Approved model metrics, or None if no approved model exists
    """

    if not LATEST_PATH.exists():
        return None

    with open(LATEST_PATH) as f:
        latest = json.load(f)

    return latest['metrics']


def compare(candidate, front) -> bool:
    """Regression gate: checks a candidate against the approved model.

    Lower is better for MAE and RMSE. The candidate passes if it does not
    worsen either metric by more than REGRESSION_TOLERANCE. With no approved
    model yet (cold start), the candidate passes.

    :param candidate: Candidate metrics dict with mae, rmse, r2
    :param front: Approved model metrics dict, or None if none exists
    :return passed: True if the candidate passes the gate
    """

    # Cold start: nothing approved yet, so nothing to regress against
    if front is None:
        return True

    c_mae = candidate['mae']
    c_rmse = candidate['rmse']

    f_mae = front['mae']
    f_rmse = front['rmse']

    mae_ok = c_mae <= f_mae * (1 + REGRESSION_TOLERANCE)
    rmse_ok = c_rmse <= f_rmse * (1 + REGRESSION_TOLERANCE)

    passed = mae_ok and rmse_ok

    return passed


def promote(model_id: str) -> bool:
    """Promotes a candidate to approved if it passes the regression gate.
    :param model_id: Candidate model id to promote
    :return promoted: True if promoted, False otherwise
    """

    APPROVED_DIR.mkdir(parents=True, exist_ok=True)

    cand_meta = CANDIDATE_DIR / f'{model_id}.metadata.json'

    if not cand_meta.exists():
        raise ValueError(f'Candidate metadata not found: {model_id}. Check models/candidates/.')

    with open(cand_meta) as f:
        metadata = json.load(f)

    candidate = metadata['metrics']
    front = load_best()

    # -=-=-=-
    # Regression gate
    # -=-=-=-

    if not compare(candidate, front):
        print(f'Candidate {model_id} rejected: regressed more than tolerance vs approved model.')
        return False

    if metadata.get('mlflow_run_id'):
        with mlflow.start_run(run_id=metadata['mlflow_run_id']):
            mlflow.set_tag('promoted', 'true')
            mlflow.set_tag('promoted_at', datetime.now(UTC).isoformat())

    # -=-=-=-
    # Copy artifact files to approved dir
    # -=-=-=-

    artifact_paths = metadata.get('artifacts', {})

    if not artifact_paths:
        raise ValueError(f'Candidate {model_id} has no artifacts listed in metadata.')

    approved_paths = {}

    for kind, src in artifact_paths.items():
        src_path = Path(src)

        if not src_path.exists():
            raise ValueError(f'Candidate artifact missing on disk: {src_path}')

        dst_path = APPROVED_DIR / src_path.name
        shutil.copy2(src_path, dst_path)
        approved_paths[kind] = str(dst_path)

    metadata['status'] = 'approved'
    metadata['artifacts'] = approved_paths

    appr_meta = APPROVED_DIR / f'{model_id}.metadata.json'

    with open(appr_meta, 'w') as f:
        json.dump(metadata, f, indent=2)

    # -=-=-=-
    # Update latest pointer
    # -=-=-=-

    pointer = {
        'model_id': model_id,
        'model_name': metadata.get('model_name'),
        'promoted_at': datetime.now(UTC).strftime('%Y-%m-%dT%H%M%SZ'),
        'metrics': candidate,
    }

    with open(LATEST_PATH, 'w') as f:
        json.dump(pointer, f, indent=2)

    print(f'Promoted model to approved: {model_id}')

    return True


def main() -> None:

    parser = argparse.ArgumentParser(description='Promote a candidate model to approved.')
    parser.add_argument('--model-id', required=True, help='Candidate model id to promote')
    args = parser.parse_args()

    promoted = promote(args.model_id)

    if not promoted:
        raise SystemExit(1)


def get_approved_metadata():
    """Returns the metadata of the currently approved model.
    :return metadata: Approved metadata dict, or None if none exists
    """

    if not LATEST_PATH.exists():
        return None

    with open(LATEST_PATH) as f:
        pointer = json.load(f)

    meta_path = APPROVED_DIR / f'{pointer["model_id"]}.metadata.json'

    if not meta_path.exists():
        return None

    with open(meta_path) as f:
        metadata = json.load(f)

    return metadata


def load_model():
    """Loads the currently approved model and its metadata for serving.

    Sklearn pipelines load from a single joblib file. Keras models load
    from a .keras file with a separate preprocessor joblib.

    Only loads self-generated, approved artifacts. joblib is pickle-based
    and can execute arbitrary code on load, so never read untrusted files.

    :return model: Object with .predict(X), or None if no approved model
    :return metadata: Approved metadata dict, or None
    """

    metadata = get_approved_metadata()

    if metadata is None:
        return None, None

    artifact_paths = metadata['artifacts']
    model_name = metadata['model_name']

    if model_name == 'nn':
        from tensorflow import keras

        keras_model = keras.models.load_model(artifact_paths['model'])
        preprocessor = joblib.load(artifact_paths['preprocessor'])

        def predict(X):
            X_t = preprocessor.transform(X)
            if hasattr(X_t, 'toarray'):
                X_t = X_t.toarray()
            y_pred_log = keras_model.predict(X_t, verbose=0).flatten()
            return np.expm1(y_pred_log)

        # Bundle the predict function on a simple namespace so the API
        # can still call model.predict(X) uniformly
        model = type('NNModel', (), {'predict': staticmethod(predict)})()

    else:
        model = joblib.load(artifact_paths['model'])

    return model, metadata


if __name__ == '__main__':
    main()
