"""
test_loss_attribution_api.py
-------------------------------
POST /losses/attribute exercised end to end (roadmap P2.4) -- proving the
engine in src/attribution/loss.py is actually wired into serving,
including the day-level soiling check that only kicks in once enough
calendar days are supplied.

RUN WITH:
  pytest tests/test_loss_attribution_api.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.pipeline import clear_sky_ghi
from src.utils.config_loader import get_plant_config

PLANT = get_plant_config("jaipur_100mw")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from src.api.main import app
    return TestClient(app)


def _clear_sky_readings(days: int = 1, day_scale: list[float] | None = None) -> list[dict]:
    """Readings for `days` full clear days at 15-min resolution, actual
    power set to exactly `day_scale[d]` times the clear-sky expected
    output for day d (default: scale 1.0 everywhere, i.e. zero loss)."""
    idx = pd.date_range("2024-06-01", periods=96 * days, freq="15min", tz="Asia/Kolkata")
    location = PLANT["location"]
    ghi = clear_sky_ghi(idx, location["latitude"], location["longitude"],
                         location.get("elevation_m", 0.0))
    capacity = PLANT["capacity"]
    expected = (ghi / 1000 * capacity["performance_ratio"] * capacity["ac_capacity_mw"]).clip(0, capacity["ac_capacity_mw"])

    if day_scale is None:
        day_scale = [1.0] * days
    day_of = idx.normalize()
    unique_days = day_of.unique()
    scale = pd.Series(0.0, index=idx)
    for day, s in zip(unique_days, day_scale):
        scale[day_of == day] = s
    actual = expected * scale

    return [
        {
            "timestamp": ts.isoformat(),
            "power_mw": float(a),
            "ghi_w_m2": float(g),
            "temperature_c": 25.0,
        }
        for ts, a, g in zip(idx, actual, ghi)
    ]


class TestLossAttributionEndpoint:
    def test_clean_day_has_no_material_loss_blocks(self, client):
        readings = _clear_sky_readings(days=1)
        resp = client.post("/losses/attribute", json={"plant_id": "jaipur_100mw", "readings": readings})
        assert resp.status_code == 200
        data = resp.json()
        assert data["material_loss_blocks"] == 0
        assert data["causes_summary"] == {}
        assert data["blocks"] == []

    def test_sustained_offset_run_is_reported_as_equipment(self, client):
        readings = _clear_sky_readings(days=1)
        # Blocks 40-43 (10:00-10:45): drop actual by a constant 15 MW,
        # same construction as tests/test_loss_attribution.py's equipment
        # scenario -- a constant offset preserves the underlying ramp's
        # shape, which the engine reads as a hardware-style loss.
        for i in range(40, 44):
            readings[i]["power_mw"] -= 15.0

        resp = client.post("/losses/attribute", json={"plant_id": "jaipur_100mw", "readings": readings})
        assert resp.status_code == 200
        data = resp.json()
        assert data["material_loss_blocks"] == 4
        assert "equipment" in data["causes_summary"]
        assert data["causes_summary"]["equipment"]["count"] == 4
        for block in data["blocks"]:
            assert block["cause"] == "equipment"
            assert block["confidence"] == "medium"
            assert "no per-inverter" in block["evidence"]["reason"]

    def test_unknown_plant_returns_404(self, client):
        readings = _clear_sky_readings(days=1)[:1]
        resp = client.post("/losses/attribute", json={"plant_id": "does_not_exist", "readings": readings})
        assert resp.status_code == 404

    def test_empty_readings_list_is_rejected(self, client):
        resp = client.post("/losses/attribute", json={"plant_id": "jaipur_100mw", "readings": []})
        assert resp.status_code == 422

    def test_mixed_aware_and_naive_timestamps_returns_422_not_500(self, client):
        readings = _clear_sky_readings(days=1)
        readings[5]["timestamp"] = readings[5]["timestamp"].replace("+05:30", "")
        resp = client.post("/losses/attribute", json={"plant_id": "jaipur_100mw", "readings": readings})
        assert resp.status_code == 422


class TestSoilingCheckViaEndpoint:
    def test_fewer_than_ten_days_reports_no_soiling_finding(self, client):
        readings = _clear_sky_readings(days=5, day_scale=[1.0, 0.95, 0.9, 0.85, 0.8])
        resp = client.post("/losses/attribute", json={"plant_id": "jaipur_100mw", "readings": readings})
        assert resp.status_code == 200
        data = resp.json()
        assert data["days_observed_for_soiling_check"] == 5
        assert data["soiling"] is None

    def test_ten_plus_days_of_steady_decline_reports_soiling(self, client):
        day_scale = list(np.linspace(1.0, 0.85, 12))
        readings = _clear_sky_readings(days=12, day_scale=day_scale)
        resp = client.post("/losses/attribute", json={"plant_id": "jaipur_100mw", "readings": readings})
        assert resp.status_code == 200
        data = resp.json()
        assert data["days_observed_for_soiling_check"] == 12
        assert data["soiling"] is not None
        assert data["soiling"]["cause"] == "soiling"
        assert data["soiling"]["slope_pct_per_day"] < -0.1
