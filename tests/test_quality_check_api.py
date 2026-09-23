"""
test_quality_check_api.py
----------------------------
POST /quality/check exercised end to end (roadmap P2.2) -- proving the
gate in src/quality/gate.py is actually wired into serving.

RUN WITH:
  pytest tests/test_quality_check_api.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from src.api.main import app
    return TestClient(app)


def _reading(timestamp: str, power_mw: float) -> dict:
    return {"timestamp": timestamp, "power_mw": power_mw}


class TestQualityCheckEndpoint:
    def test_clean_readings_pass(self, client):
        # A real diurnal curve, not a flat step -- a constant midday value
        # for several hours in a row would itself be a flatline by design
        # (see src/quality/gate.py), so this uses varying values to
        # isolate "genuinely clean data passes" from that check.
        hours = np.arange(24)
        solar_factor = np.clip(np.cos((hours - 12.5) * np.pi / 12), 0, None)
        readings = [_reading(f"2024-06-01T{h:02d}:00:00+05:30", float(70 * f))
                    for h, f in zip(hours, solar_factor)]
        resp = client.post("/quality/check", json={"plant_id": "jaipur_100mw", "readings": readings})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_rows"] == 24
        assert data["passed"] is True
        assert data["issues"] == []

    def test_night_time_non_zero_is_reported(self, client):
        readings = [_reading(f"2024-06-01T{h:02d}:00:00+05:30", 0.0) for h in range(24)]
        readings[2]["power_mw"] = 15.0  # 02:00 -- unambiguous night
        resp = client.post("/quality/check", json={"plant_id": "jaipur_100mw", "readings": readings})
        assert resp.status_code == 200
        data = resp.json()
        assert data["passed"] is False
        checks = {i["check"] for i in data["issues"]}
        assert "night_time_non_zero" in checks

    def test_missing_timezone_is_reported(self, client):
        readings = [_reading(f"2024-06-01T{h:02d}:00:00", 0.0) for h in range(24)]  # no offset
        resp = client.post("/quality/check", json={"plant_id": "jaipur_100mw", "readings": readings})
        assert resp.status_code == 200
        data = resp.json()
        assert data["passed"] is False
        checks = {i["check"] for i in data["issues"]}
        assert "missing_timezone" in checks

    def test_unknown_plant_returns_404(self, client):
        readings = [_reading("2024-06-01T12:00:00+05:30", 10.0)]
        resp = client.post("/quality/check", json={"plant_id": "does_not_exist", "readings": readings})
        assert resp.status_code == 404

    def test_empty_readings_list_is_rejected(self, client):
        resp = client.post("/quality/check", json={"plant_id": "jaipur_100mw", "readings": []})
        assert resp.status_code == 422
