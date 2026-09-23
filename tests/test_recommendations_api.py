"""
test_recommendations_api.py
------------------------------
POST /recommendations/generate, GET /recommendations, and
POST /recommendations/{id}/decide exercised end to end (roadmap P2.6) --
proving src/recommendations/engine.py and store.py are actually wired
into serving. Every test points the API at one shared in-memory SQLite
connection (via monkeypatching get_recommendations_db) so state persists
across calls WITHIN a test but never touches the real on-disk log or
leaks between tests.

RUN WITH:
  pytest tests/test_recommendations_api.py -v
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.recommendations import store as recommendations_store


@pytest.fixture
def client(monkeypatch):
    from fastapi.testclient import TestClient
    import src.api.main as main_module

    shared_conn = recommendations_store.connect(":memory:")
    monkeypatch.setattr(main_module, "get_recommendations_db", lambda: shared_conn)
    return TestClient(main_module.app)


class TestGenerateFromScheduleRisk:
    def test_high_risk_block_generates_a_schedule_revision_recommendation(self, client):
        resp = client.post("/recommendations/generate", json={
            "plant_id": "jaipur_100mw",
            "schedule_risk": {
                "declared_schedule_mw": [50],
                "p10_mw": [48], "p50_mw": [50], "p90_mw": [70],
                "as_of": "2024-06-01",
            },
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["generated"] == 1
        rec = data["recommendations"][0]
        assert rec["recommendation_type"] == "schedule_revision"
        assert rec["status"] == "pending"
        assert rec["plant_id"] == "jaipur_100mw"

    def test_medium_risk_block_with_battery_state_generates_a_battery_action(self, client):
        resp = client.post("/recommendations/generate", json={
            "plant_id": "jaipur_100mw",
            "schedule_risk": {
                "declared_schedule_mw": [50],
                "p10_mw": [36], "p50_mw": [38], "p90_mw": [50],
                "as_of": "2024-06-01",
            },
            "battery_state": {
                "soc_mwh": 10.0, "capacity_mwh": 20.0,
                "charge_rate_mw": 25.0, "discharge_rate_mw": 25.0,
            },
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["generated"] == 1
        assert data["recommendations"][0]["recommendation_type"] == "battery_action"

    def test_low_risk_block_generates_nothing(self, client):
        resp = client.post("/recommendations/generate", json={
            "plant_id": "jaipur_100mw",
            "schedule_risk": {
                "declared_schedule_mw": [50],
                "p10_mw": [49], "p50_mw": [50], "p90_mw": [51],
                "as_of": "2024-06-01",
            },
        })
        assert resp.status_code == 200
        assert resp.json()["generated"] == 0

    def test_mismatched_lengths_are_rejected(self, client):
        resp = client.post("/recommendations/generate", json={
            "plant_id": "jaipur_100mw",
            "schedule_risk": {
                "declared_schedule_mw": [50, 50],
                "p10_mw": [48], "p50_mw": [50, 50], "p90_mw": [70, 70],
            },
        })
        assert resp.status_code == 422


class TestGenerateFromLossReadings:
    def _clear_day_readings(self):
        import numpy as np
        import pandas as pd
        from src.features.pipeline import clear_sky_ghi
        from src.utils.config_loader import get_plant_config

        plant = get_plant_config("jaipur_100mw")
        idx = pd.date_range("2024-06-01", periods=96, freq="15min", tz="Asia/Kolkata")
        location = plant["location"]
        ghi = clear_sky_ghi(idx, location["latitude"], location["longitude"], location.get("elevation_m", 0.0))
        capacity = plant["capacity"]
        expected = (ghi / 1000 * capacity["performance_ratio"] * capacity["ac_capacity_mw"]).clip(0, capacity["ac_capacity_mw"])
        return idx, ghi, expected

    def test_equipment_run_generates_one_inspection_recommendation(self, client):
        idx, ghi, expected = self._clear_day_readings()
        actual = expected.copy()
        for i in range(40, 44):
            actual.iloc[i] -= 15.0
        readings = [
            {"timestamp": ts.isoformat(), "power_mw": float(a), "ghi_w_m2": float(g), "temperature_c": 25.0}
            for ts, a, g in zip(idx, actual, ghi)
        ]
        resp = client.post("/recommendations/generate", json={
            "plant_id": "jaipur_100mw",
            "loss_readings": readings,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["generated"] == 1
        assert data["recommendations"][0]["recommendation_type"] == "inspection"

    def test_unknown_plant_returns_404(self, client):
        resp = client.post("/recommendations/generate", json={"plant_id": "does_not_exist"})
        assert resp.status_code == 404


class TestListingAndDeciding:
    def _generate_one(self, client):
        resp = client.post("/recommendations/generate", json={
            "plant_id": "jaipur_100mw",
            "schedule_risk": {
                "declared_schedule_mw": [50],
                "p10_mw": [48], "p50_mw": [50], "p90_mw": [70],
                "as_of": "2024-06-01",
            },
        })
        return resp.json()["recommendations"][0]["id"]

    def test_generated_recommendation_shows_up_in_the_listing(self, client):
        rec_id = self._generate_one(client)
        resp = client.get("/recommendations")
        assert resp.status_code == 200
        ids = [r["id"] for r in resp.json()["recommendations"]]
        assert rec_id in ids

    def test_approve_then_filter_by_status(self, client):
        rec_id = self._generate_one(client)
        resp = client.post(f"/recommendations/{rec_id}/decide", json={
            "decision": "approved", "decided_by": "ops_alice",
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "approved"

        approved = client.get("/recommendations", params={"status": "approved"}).json()["recommendations"]
        assert any(r["id"] == rec_id for r in approved)
        pending = client.get("/recommendations", params={"status": "pending"}).json()["recommendations"]
        assert not any(r["id"] == rec_id for r in pending)

    def test_deciding_an_unknown_id_returns_404(self, client):
        resp = client.post("/recommendations/999999/decide", json={
            "decision": "approved", "decided_by": "ops_alice",
        })
        assert resp.status_code == 404

    def test_invalid_decision_value_returns_422(self, client):
        rec_id = self._generate_one(client)
        resp = client.post(f"/recommendations/{rec_id}/decide", json={
            "decision": "maybe", "decided_by": "ops_alice",
        })
        assert resp.status_code == 422
