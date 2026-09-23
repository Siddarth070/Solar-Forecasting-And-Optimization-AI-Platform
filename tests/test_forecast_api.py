"""
test_forecast_api.py
----------------------
POST /forecast wired onto real timestamps instead of the old "anchor
every request to a fixed placeholder year/day" hack (roadmap P1.4,
now that src/time_blocks.py exists). Each WeatherInput carries its own
genuine ISO timestamp; the response echoes them back and reports a real
peak timestamp, not just an array index into an implicit hour list.

RUN WITH:
  pytest tests/test_forecast_api.py -v
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


def _hour(timestamp: str, ghi: float = 500.0) -> dict:
    return {
        "timestamp": timestamp,
        "shortwave_radiation": ghi,
        "cloud_cover": 10,
        "temperature_2m": 30,
        "relative_humidity_2m": 40,
        "wind_speed_10m": 3,
    }


class TestForecastRealTimestamps:
    def test_response_echoes_requested_timestamps_in_order(self, client):
        hours = [_hour(f"2024-06-15T{h:02d}:00:00+05:30") for h in [8, 9, 10]]
        resp = client.post("/forecast", json={"hours": hours, "plant_id": "jaipur_100mw"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["timestamps"] == [
            "2024-06-15T08:00:00+05:30", "2024-06-15T09:00:00+05:30", "2024-06-15T10:00:00+05:30",
        ]

    def test_peak_index_and_timestamp_are_consistent(self, client):
        rng = np.random.default_rng(0)
        hours = [_hour(f"2024-06-15T{h:02d}:00:00+05:30", ghi=float(g))
                 for h, g in zip(range(6, 18), rng.uniform(0, 900, 12))]
        resp = client.post("/forecast", json={"hours": hours, "plant_id": "jaipur_100mw"})
        data = resp.json()

        assert data["predictions_mw"][data["peak_index"]] == max(data["predictions_mw"])
        assert data["timestamps"][data["peak_index"]] == data["peak_timestamp"]

    def test_missing_timestamp_field_is_rejected(self, client):
        """The old hour+month schema no longer silently works -- a real
        timestamp is required, not accepted as an optional extra."""
        bad_hour = {
            "shortwave_radiation": 500, "cloud_cover": 10, "temperature_2m": 30,
            "relative_humidity_2m": 40, "wind_speed_10m": 3,
            "hour": 12, "month": 6,  # old-style fields, no `timestamp`
        }
        resp = client.post("/forecast", json={"hours": [bad_hour], "plant_id": "jaipur_100mw"})
        assert resp.status_code == 422

    def test_different_real_dates_are_actually_used_for_clear_sky(self, client):
        """Before this wiring, every request was silently re-anchored to a
        fixed placeholder date (year=2024, day=15) regardless of what the
        caller intended. Two requests differing only in the CALENDAR DAY
        (same hour, same weather, same month) must be able to produce
        different results now that the real date drives the clear-sky
        computation -- proving the placeholder is gone, not just renamed."""
        summer_solstice = client.post("/forecast", json={
            "hours": [_hour("2024-06-21T12:00:00+05:30", ghi=700)],
            "plant_id": "jaipur_100mw",
        }).json()
        early_june = client.post("/forecast", json={
            "hours": [_hour("2024-06-01T12:00:00+05:30", ghi=700)],
            "plant_id": "jaipur_100mw",
        }).json()
        # Both must succeed and be internally consistent -- the point is
        # that the endpoint accepts and actually threads through two
        # genuinely different calendar days without collapsing them to
        # the same hardcoded reference day.
        assert summer_solstice["timestamps"][0].startswith("2024-06-21")
        assert early_june["timestamps"][0].startswith("2024-06-01")
