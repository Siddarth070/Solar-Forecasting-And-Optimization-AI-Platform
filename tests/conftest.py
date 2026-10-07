"""Suite-wide defaults: the API's write endpoints are closed and POSTs are
rate-limited unless configured otherwise (see src/api/main.py's public-
deployment guards). Most tests exercise endpoint behavior, not those
guards, so open writes and lift the limit here; tests/test_public_guards.py
re-closes them to test the guards themselves."""

import pytest


@pytest.fixture(autouse=True)
def _local_dev_api_env(monkeypatch):
    monkeypatch.delenv("ZENITH_WRITE_API_KEY", raising=False)
    monkeypatch.setenv("ZENITH_ALLOW_OPEN_WRITES", "1")
    monkeypatch.setenv("ZENITH_RATE_LIMIT_PER_MIN", "0")
