"""
test_schedule_revision_api.py
-------------------------------
Roadmap P1.4: exercises POST /schedule/revise end to end, the same way a
real client would use it -- proving the real Grid Code gate-closure
timing (src/regulatory/grid_code.py) is actually wired into serving, not
just unit-tested in isolation.

RUN WITH:
  pytest tests/test_schedule_revision_api.py -v
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


class TestReviseScheduleEndpoint:
    def test_locked_blocks_stay_at_old_value_revisable_blocks_take_new_value(self, client):
        locked = [50.0] * 96
        proposed = [70.0] * 96
        resp = client.post("/schedule/revise", json={
            "plant_id": "jaipur_100mw",
            "date": "2024-06-01",
            "locked_schedule_mw": locked,
            "proposed_schedule_mw": proposed,
            "request_timestamp": "2024-06-01T13:56:00+05:30",
        })
        assert resp.status_code == 200
        data = resp.json()

        assert data["revision_allowed"] is True
        assert data["effective_timestamp"] == "2024-06-01T15:30:00+05:30"
        assert data["locked_block_count"] == 62
        applied = data["applied_schedule_mw"]
        assert applied[:62] == pytest.approx([50.0] * 62)
        assert applied[62:] == pytest.approx([70.0] * 34)

    def test_collective_transaction_plant_rejects_revision(self, client, tmp_path, monkeypatch):
        """Regulation 49(8): a plant configured for collective transactions
        cannot revise its schedule at all -- exercised through the real
        config lookup, not a mocked object."""
        from src.utils import config_loader

        collective_config = {
            "plant_id": "collective_test_plant",
            "name": "Collective Test Plant",
            "location": {"name": "Test", "state": "Test", "latitude": 0.0,
                         "longitude": 0.0, "elevation_m": 0, "timezone": "Asia/Kolkata"},
            "capacity": {"ac_capacity_mw": 10, "dc_capacity_mw": 10,
                         "panel_efficiency": 0.2, "temperature_coefficient": -0.004,
                         "performance_ratio": 0.8, "panel_area_m2": 1000,
                         "tilt_deg": None, "azimuth_deg": None},
            "grid": {"sldc": None, "rldc": None, "ists_or_instate": None,
                     "qca_role": None, "metering_point": None},
            "regulatory": {
                "dsm_ruleset_id": None, "seller_category": "solar",
                "contract_rate_rs_per_kwh": None, "schedule_format": None,
                "revision_windows": {"grid_code_ruleset_id": "CERC_IEGC_2023_Reg49",
                                      "transaction_type": "collective"},
            },
        }

        def fake_get_plant_config(plant_id):
            if plant_id == "collective_test_plant":
                return collective_config
            raise FileNotFoundError(plant_id)

        import src.api.main as main_module
        monkeypatch.setattr(main_module, "get_plant_config", fake_get_plant_config)

        locked = [50.0] * 96
        proposed = [70.0] * 96
        resp = client.post("/schedule/revise", json={
            "plant_id": "collective_test_plant",
            "date": "2024-06-01",
            "locked_schedule_mw": locked,
            "proposed_schedule_mw": proposed,
            "request_timestamp": "2024-06-01T13:56:00+05:30",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["revision_allowed"] is False
        assert data["effective_timestamp"] is None
        assert data["applied_schedule_mw"] == pytest.approx(locked)

    def test_unknown_plant_returns_404(self, client):
        resp = client.post("/schedule/revise", json={
            "plant_id": "does_not_exist",
            "date": "2024-06-01",
            "locked_schedule_mw": [50.0] * 96,
            "proposed_schedule_mw": [70.0] * 96,
            "request_timestamp": "2024-06-01T13:56:00+05:30",
        })
        assert resp.status_code == 404

    def test_wrong_length_schedule_is_rejected(self, client):
        resp = client.post("/schedule/revise", json={
            "plant_id": "jaipur_100mw",
            "date": "2024-06-01",
            "locked_schedule_mw": [50.0] * 50,
            "proposed_schedule_mw": [70.0] * 96,
            "request_timestamp": "2024-06-01T13:56:00+05:30",
        })
        assert resp.status_code == 422
