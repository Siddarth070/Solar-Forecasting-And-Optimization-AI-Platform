"""
test_schedule_risk_api.py
----------------------------
POST /schedule/risk exercised end to end (roadmap P2.5) -- proving the
engine in src/risk/schedule_risk.py is actually wired into serving,
including the real 96-block grid enforcement and the as_of/date DSM
cutover resolution.

RUN WITH:
  pytest tests/test_schedule_risk_api.py -v
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from src.api.main import app
    return TestClient(app)


class TestScheduleRiskEndpoint:
    def test_low_risk_block_reports_disclaimer_and_band(self, client):
        resp = client.post("/schedule/risk", json={
            "plant_id": "jaipur_100mw",
            "declared_schedule_mw": [50],
            "p10_mw": [48], "p50_mw": [50], "p90_mw": [52],
            "dt_hours": 0.25,
            "as_of": "2024-06-01",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["blocks"] == 1
        assert data["risk_summary"] == {"low": 1, "medium": 0, "high": 0}
        assert data["disclaimer"] == (
            "Indicative DSM exposure based on configured assumptions and uploaded data. "
            "Not an official settlement statement."
        )
        block = data["schedule"][0]
        assert block["risk_level"] == "low"
        assert block["disclaimer"] == data["disclaimer"]

    def test_wide_tail_spread_reports_high_risk(self, client):
        resp = client.post("/schedule/risk", json={
            "plant_id": "jaipur_100mw",
            "declared_schedule_mw": [50],
            "p10_mw": [48], "p50_mw": [50], "p90_mw": [70],
            "dt_hours": 0.25,
            "as_of": "2024-06-01",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["schedule"][0]["risk_level"] == "high"
        assert data["schedule"][0]["worst_band"] == 3

    def test_date_given_enforces_the_96_block_grid_and_returns_timestamps(self, client):
        resp = client.post("/schedule/risk", json={
            "plant_id": "jaipur_100mw",
            "declared_schedule_mw": [50] * 50,  # wrong length on purpose
            "p10_mw": [48] * 50, "p50_mw": [50] * 50, "p90_mw": [52] * 50,
            "date": "2024-06-01",
        })
        assert resp.status_code == 422
        assert "96" in resp.json()["detail"]

    def test_correct_96_block_length_with_date_returns_real_timestamps(self, client):
        resp = client.post("/schedule/risk", json={
            "plant_id": "jaipur_100mw",
            "declared_schedule_mw": [50] * 96,
            "p10_mw": [48] * 96, "p50_mw": [50] * 96, "p90_mw": [52] * 96,
            "date": "2024-06-01",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["blocks"] == 96
        assert data["schedule"][0]["timestamp"] == "2024-06-01T00:00:00+05:30"
        assert data["schedule"][95]["timestamp"] == "2024-06-01T23:45:00+05:30"

    def test_mismatched_lengths_without_date_are_rejected(self, client):
        resp = client.post("/schedule/risk", json={
            "plant_id": "jaipur_100mw",
            "declared_schedule_mw": [50, 50],
            "p10_mw": [48], "p50_mw": [50, 50], "p90_mw": [52, 52],
        })
        assert resp.status_code == 422

    def test_unknown_plant_returns_404(self, client):
        resp = client.post("/schedule/risk", json={
            "plant_id": "does_not_exist",
            "declared_schedule_mw": [50], "p10_mw": [48], "p50_mw": [50], "p90_mw": [52],
        })
        assert resp.status_code == 404

    def test_plant_missing_dsm_config_returns_422_not_500(self, client, monkeypatch):
        import src.api.main as main_module

        def fake_get_plant_config(plant_id):
            return {
                "plant_id": plant_id,
                "capacity": {"ac_capacity_mw": 100},
                "regulatory": {},  # no seller_category / contract_rate_rs_per_kwh
            }

        monkeypatch.setattr(main_module, "get_plant_config", fake_get_plant_config)
        resp = client.post("/schedule/risk", json={
            "plant_id": "jaipur_100mw",
            "declared_schedule_mw": [50], "p10_mw": [48], "p50_mw": [50], "p90_mw": [52],
        })
        assert resp.status_code == 422
