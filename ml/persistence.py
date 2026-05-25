"""persistence.py: Promotes candidate models to approved and maintains the latest pointer."""

__author__ = 'Majd Jamal'

import argparse
import json
import shutil
from datetime import UTC, datetime

import joblib

from ml.config import APPROVED_DIR, CANDIDATE_DIR

LATEST_PATH = APPROVED_DIR / 'latest.json'
REGRESSION_TOLERANCE = 0.10


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

    Locates the candidate files, runs the gate against the currently approved
    model, and on pass copies the artifact and metadata into the approved
    directory, flips status to 'approved', and repoints latest.json.

    :param model_id: Candidate model id to promote
    :return promoted: True if promoted, False otherwise
    """

    APPROVED_DIR.mkdir(parents=True, exist_ok=True)

    cand_model = CANDIDATE_DIR / f'{model_id}.joblib'
    cand_meta = CANDIDATE_DIR / f'{model_id}.metadata.json'

    if not cand_model.exists() or not cand_meta.exists():
        raise ValueError(f'Candidate not found: {model_id}. Check models/candidates/.')

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

    # -=-=-=-
    # Promote artifact and metadata
    # -=-=-=-

    metadata['status'] = 'approved'

    appr_model = APPROVED_DIR / f'{model_id}.joblib'
    appr_meta = APPROVED_DIR / f'{model_id}.metadata.json'

    shutil.copy2(cand_model, appr_model)

    with open(appr_meta, 'w') as f:
        json.dump(metadata, f, indent=2)

    # -=-=-=-
    # Update latest pointer
    # -=-=-=-

    pointer = {
        'model_id': model_id,
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


def get_approved_model_path():
    """Returns the artifact path of the currently approved model.
    :return model_path: Path to the approved .joblib, or None if none exists
    """

    if not LATEST_PATH.exists():
        return None

    with open(LATEST_PATH) as f:
        latest = json.load(f)

    model_path = APPROVED_DIR / f'{latest["model_id"]}.joblib'

    return model_path


def load_model():
    """Loads the currently approved model and its metadata for serving.

    Only loads self-generated, approved artifacts. joblib is pickle-based and
    can execute arbitrary code on load, so never read untrusted files.

    :return model: Fitted pipeline, or None if no approved model exists
    :return metadata: Approved model metadata dict, or None if none exists
    """

    model_path = get_approved_model_path()

    if model_path is None:
        return None, None

    with open(LATEST_PATH) as f:
        model_id = json.load(f)['model_id']

    meta_path = APPROVED_DIR / f'{model_id}.metadata.json'

    model = joblib.load(model_path)

    with open(meta_path) as f:
        metadata = json.load(f)

    return model, metadata


if __name__ == '__main__':
    main()
