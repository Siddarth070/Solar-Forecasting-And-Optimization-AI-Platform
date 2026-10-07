"""
test_optimize_api.py
----------------------
POST /optimize's real 96-block/15-minute grid enforcement (roadmap P1.4):
when `date` is given, solar_forecast_mw/declared_schedule_mw must align
to the real grid (src/time_blocks.py) -- exactly 96 entries -- and
dt_hours is forced to 0.25 regardless of what the client passed, with
real block-start timestamps returned. Omitting `date` keeps the
endpoint's original, unconstrained behavior (any length, caller's own
dt_hours) -- a deliberate backward-compatible default, not an oversight.

RUN WITH:
  pytest tests/test_optimize_api.py -v
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


class TestOptimizeWithoutDate:
    def test_arbitrary_length_and_custom_dt_hours_still_work(self, client):
        resp = client.post("/optimize", json={
            "solar_forecast_mw": [10.0] * 10,
            "declared_schedule_mw": [8.0] * 10,
            "dt_hours": 1.0,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["blocks"] == 10
        assert "block_start" not in data["schedule"][0]


class TestOptimizeWithDate:
    def test_wrong_length_is_rejected_with_a_clear_message(self, client):
        resp = client.post("/optimize", json={
            "solar_forecast_mw": [10.0] * 50,
            "declared_schedule_mw": [8.0] * 50,
            "date": "2024-06-01",
        })
        assert resp.status_code == 422
        assert "96" in resp.json()["detail"]

    def test_correct_length_returns_real_block_start_timestamps(self, client):
        resp = client.post("/optimize", json={
            "solar_forecast_mw": [10.0] * 96,
            "declared_schedule_mw": [8.0] * 96,
            "date": "2024-06-01",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["blocks"] == 96
        assert data["schedule"][0]["block_start"] == "2024-06-01T00:00:00+05:30"
        assert data["schedule"][1]["block_start"] == "2024-06-01T00:15:00+05:30"
        assert data["schedule"][95]["block_start"] == "2024-06-01T23:45:00+05:30"

    def test_dt_hours_is_forced_to_quarter_hour_regardless_of_request(self, client):
        """A client passing an inconsistent dt_hours alongside `date` must
        not silently get a physically wrong result -- the endpoint always
        runs at the real grid's 0.25h, and the two calls below (differing
        ONLY in the client-supplied dt_hours) must therefore agree
        exactly."""
        rng = np.random.default_rng(1)
        solar = rng.uniform(0, 80, 96).tolist()
        schedule = rng.uniform(0, 60, 96).tolist()

        resp_a = client.post("/optimize", json={
            "solar_forecast_mw": solar, "declared_schedule_mw": schedule,
            "date": "2024-06-01", "dt_hours": 1.0,  # wrong on purpose
        })
        resp_b = client.post("/optimize", json={
            "solar_forecast_mw": solar, "declared_schedule_mw": schedule,
            "date": "2024-06-01", "dt_hours": 0.25,  # matches the real grid
        })
        assert resp_a.status_code == resp_b.status_code == 200
        assert resp_a.json()["summary"] == resp_b.json()["summary"]

    def test_unknown_plant_id_still_returns_404_with_date_given(self, client):
        resp = client.post("/optimize", json={
            "solar_forecast_mw": [10.0] * 96,
            "declared_schedule_mw": [8.0] * 96,
            "date": "2024-06-01",
            "plant_id": "does_not_exist",
        })
        assert resp.status_code == 404

    def test_plant_with_export_limit_gets_curtailment_enforced_end_to_end(self, client, monkeypatch):
        # Roadmap P2.3: grid.export_limit_mw, once configured, must be
        # enforced by /optimize -- not just stored.
        import src.api.main as main_module

        def fake_get_plant_config(plant_id):
            return {
                "plant_id": plant_id,
                "capacity": {"ac_capacity_mw": 100},
                "grid": {"export_limit_mw": 20.0},
                "regulatory": {},
            }

        monkeypatch.setattr(main_module, "get_plant_config", fake_get_plant_config)
        resp = client.post("/optimize", json={
            "solar_forecast_mw": [80.0],
            "declared_schedule_mw": [10.0],
            "plant_id": "capped_plant",
            "battery_capacity_mwh": 20, "charge_rate_mw": 5, "discharge_rate_mw": 5,
            "initial_charge_mwh": 10, "dt_hours": 1.0, "round_trip_efficiency": 1.0,
        })
        assert resp.status_code == 200
        record = resp.json()["schedule"][0]
        grid_delivered = record["solar_mw"] - record["curtailed_mw"] + record["discharge_mw"] - record["charge_mw"]
        assert grid_delivered <= 20.0 + 1e-6
        assert record["curtailed_mw"] > 0
