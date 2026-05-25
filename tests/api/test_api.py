"""test_api.py: Contract tests for the Random Houses API."""

__author__ = 'Majd Jamal'


def test_health_ok(client):
    assert client.get('/health').status_code == 200


def test_predict_returns_price(client, payload):
    r = client.post('/predict', json=payload)
    assert r.status_code == 200, f'Expected 200, got {r.status_code} — is a model promoted?'

    body = r.json()
    assert 'predicted_price' in body
    assert body['predicted_price'] > 0  # sanity: a price, not garbage
    assert 'model_version' in body  # traceability field is present


def test_predict_rejects_bad_input(client):
    r = client.post('/predict', json={'rooms': 'three'})
    assert r.status_code == 422
