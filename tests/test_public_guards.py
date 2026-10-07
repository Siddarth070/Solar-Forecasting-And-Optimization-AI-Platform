"""
test_public_guards.py
---------------------
The guards that make the API safe to expose publicly (src/api/main.py):
write endpoints closed unless ZENITH_WRITE_API_KEY or
ZENITH_ALLOW_OPEN_WRITES is set, an X-API-Key check, a request-body cap,
per-client POST rate limiting, and per-block list-length caps.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

WRITE_CALLS = [
    ("/plants", {}),
    ("/recommendations/generate", {}),
    ("/recommendations/1/decide", {}),
]


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from src.api.main import app
    return TestClient(app)


@pytest.fixture
def production_env(monkeypatch):
    """What a deployment looks like if nobody configured writes."""
    monkeypatch.delenv("ZENITH_ALLOW_OPEN_WRITES", raising=False)
    monkeypatch.delenv("ZENITH_WRITE_API_KEY", raising=False)


class TestWriteAccess:
    @pytest.mark.parametrize("path,body", WRITE_CALLS)
    def test_writes_closed_by_default(self, client, production_env, path, body):
        resp = client.post(path, json=body)
        assert resp.status_code == 403

    @pytest.mark.parametrize("path,body", WRITE_CALLS)
    def test_key_required_when_configured(self, client, production_env, monkeypatch, path, body):
        monkeypatch.setenv("ZENITH_WRITE_API_KEY", "s3cret")
        assert client.post(path, json=body).status_code == 401
        assert client.post(path, json=body, headers={"X-API-Key": "wrong"}).status_code == 401
        # Right key passes the guard: the request then fails only on its
        # (deliberately empty) body, never on access.
        resp = client.post(path, json=body, headers={"X-API-Key": "s3cret"})
        assert resp.status_code not in (401, 403)

    def test_key_wins_over_open_writes_flag(self, client, monkeypatch):
        monkeypatch.setenv("ZENITH_ALLOW_OPEN_WRITES", "1")
        monkeypatch.setenv("ZENITH_WRITE_API_KEY", "s3cret")
        assert client.post("/plants", json={}).status_code == 401

    def test_read_and_compute_endpoints_stay_open(self, client, production_env):
        assert client.get("/plants").status_code == 200
        assert client.get("/plants/jaipur_100mw").status_code == 200
        resp = client.post("/dsm/estimate", json={"scheduled_mw": [10.0], "injected_mw": [10.0]})
        assert resp.status_code == 200


class TestLimits:
    def test_rate_limit_returns_429(self, client, monkeypatch):
        import src.api.main as api
        monkeypatch.setenv("ZENITH_RATE_LIMIT_PER_MIN", "3")
        api._rate_hits.clear()
        body = {"scheduled_mw": [10.0], "injected_mw": [10.0]}
        codes = [client.post("/dsm/estimate", json=body).status_code for _ in range(4)]
        assert codes == [200, 200, 200, 429]
        assert client.get("/plants").status_code == 200  # GETs are not limited
        api._rate_hits.clear()

    def test_forwarded_for_only_trusted_when_configured(self, client, monkeypatch):
        import src.api.main as api
        monkeypatch.setenv("ZENITH_RATE_LIMIT_PER_MIN", "1")
        body = {"scheduled_mw": [10.0], "injected_mw": [10.0]}

        api._rate_hits.clear()
        monkeypatch.setenv("ZENITH_TRUST_PROXY", "1")
        assert client.post("/dsm/estimate", json=body, headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 200
        assert client.post("/dsm/estimate", json=body, headers={"X-Forwarded-For": "2.2.2.2"}).status_code == 200

        # A client-forged first hop doesn't buy a fresh bucket: the proxy-
        # appended last hop is the key.
        assert client.post("/dsm/estimate", json=body,
                           headers={"X-Forwarded-For": "9.9.9.9, 2.2.2.2"}).status_code == 429

        api._rate_hits.clear()
        monkeypatch.delenv("ZENITH_TRUST_PROXY")
        assert client.post("/dsm/estimate", json=body, headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 200
        assert client.post("/dsm/estimate", json=body, headers={"X-Forwarded-For": "2.2.2.2"}).status_code == 429
        api._rate_hits.clear()

    def test_oversized_body_is_413(self, client):
        resp = client.post("/dsm/estimate", content=b"x" * 1_000_001,
                           headers={"Content-Type": "application/json"})
        assert resp.status_code == 413

    def test_optimize_rejects_more_than_a_week_of_blocks(self, client):
        n = 7 * 96 + 1
        resp = client.post("/optimize", json={"solar_forecast_mw": [1.0] * n, "declared_schedule_mw": [1.0] * n})
        assert resp.status_code == 422
