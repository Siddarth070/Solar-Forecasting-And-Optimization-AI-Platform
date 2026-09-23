"""
test_weekly_report_api.py
----------------------------
POST /reports/weekly exercised end to end (roadmap P2.9) -- proving
src/reporting/weekly_report.py is actually wired into serving via the
same build_features pipeline /forecast uses. The exact-arithmetic
scenarios already live in tests/test_weekly_report.py; this file checks
the HTTP wiring: shapes, status codes, and the validation paths.

RUN WITH:
  pytest tests/test_weekly_report_api.py -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from src.api.main import app
    return TestClient(app)


def _readings_and_forecasts():
    from src.features.pipeline import clear_sky_ghi
    from src.utils.config_loader import get_plant_config

    plant = get_plant_config("jaipur_100mw")
    location = plant["location"]
    capacity = plant["capacity"]

    idx = pd.date_range("2024-06-01 00:00", periods=48, freq="h", tz="Asia/Kolkata")
    ghi = clear_sky_ghi(idx, location["latitude"], location["longitude"], location.get("elevation_m", 0.0))
    actual = (ghi / 1000 * capacity["performance_ratio"] * capacity["ac_capacity_mw"]).clip(0, capacity["ac_capacity_mw"])

    readings = [
        {
            "timestamp": ts.isoformat(),
            "shortwave_radiation": float(g),
            "cloud_cover": 0.0,
            "temperature_2m": 25.0,
            "relative_humidity_2m": 40.0,
            "wind_speed_10m": 3.0,
            "solar_output_mw": float(a),
        }
        for ts, g, a in zip(idx, ghi, actual)
    ]

    # Score the second day (hours 24-47) against slightly-off predictions.
    forecasts = [
        {"target_timestamp": ts.isoformat(), "horizon_hours": 1, "predicted_mw": float(a) + 2.0}
        for ts, a in zip(idx[24:], actual[24:])
    ]
    return readings, forecasts


class TestWeeklyReportEndpoint:
    def test_returns_a_full_report_shape(self, client):
        readings, forecasts = _readings_and_forecasts()
        resp = client.post("/reports/weekly", json={
            "plant_id": "jaipur_100mw", "readings": readings, "forecasts": forecasts,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["plant_id"] == "jaipur_100mw"
        assert data["n_forecasts"] == 24
        assert set(data["overall"].keys()) == {"model", "persistence", "smart_persistence", "physics"}
        assert "1" in data["by_horizon_hours"]
        assert data["overall"]["model"]["n"] > 0

    def test_unknown_plant_returns_404(self, client):
        readings, forecasts = _readings_and_forecasts()
        resp = client.post("/reports/weekly", json={
            "plant_id": "does_not_exist", "readings": readings, "forecasts": forecasts,
        })
        assert resp.status_code == 404

    def test_empty_forecasts_is_rejected(self, client):
        readings, _ = _readings_and_forecasts()
        resp = client.post("/reports/weekly", json={
            "plant_id": "jaipur_100mw", "readings": readings, "forecasts": [],
        })
        assert resp.status_code == 422

    def test_empty_readings_is_rejected(self, client):
        _, forecasts = _readings_and_forecasts()
        resp = client.post("/reports/weekly", json={
            "plant_id": "jaipur_100mw", "readings": [], "forecasts": forecasts,
        })
        assert resp.status_code == 422

    def test_forecast_target_not_covered_by_readings_returns_422(self, client):
        readings, forecasts = _readings_and_forecasts()
        forecasts[0]["target_timestamp"] = "2030-01-01T00:00:00+05:30"
        resp = client.post("/reports/weekly", json={
            "plant_id": "jaipur_100mw", "readings": readings, "forecasts": forecasts,
        })
        assert resp.status_code == 422
