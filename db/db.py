"""db.py: SQLite persistence for prediction logging."""

__author__ = 'Majd Jamal'

import json
import os
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path

DB_PATH = Path(os.environ.get('DB_PATH', 'data/runtime/house_price.db'))


SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    request_id      TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL,
    model_version   TEXT NOT NULL,
    features        TEXT NOT NULL,
    predicted_price REAL NOT NULL
) STRICT;
"""


def connect():
    """Opens a SQLite connection to the configured database file.
    :return conn: Open sqlite3 connection with row access by name
    """

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    return conn


def init_db() -> None:
    """Creates the database file and predictions table if they do not exist.

    Called once at API startup so the table is guaranteed to exist before
    the first prediction is logged.
    """

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = connect()

    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()

    print(f'Initialized database at {DB_PATH}')


def log_prediction(features: dict, predicted_price: float, model_version: str) -> str:
    """Logs a single prediction to the database.

    :param features: Validated input features as a dict
    :param predicted_price: Predicted house price
    :param model_version: Model id that produced the prediction
    :return request_id: Generated id for this prediction
    """

    request_id = str(uuid.uuid4())
    created_at = datetime.now(UTC).isoformat()
    payload = json.dumps(features)

    conn = connect()

    try:
        conn.execute(
            'INSERT INTO predictions '
            '(request_id, created_at, model_version, features, predicted_price) '
            'VALUES (?, ?, ?, ?, ?)',
            (request_id, created_at, model_version, payload, predicted_price),
        )
        conn.commit()
    finally:
        conn.close()

    return request_id


def main() -> None:
    init_db()


if __name__ == '__main__':
    main()
