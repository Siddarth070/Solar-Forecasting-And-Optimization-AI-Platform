"""
test_plant_registration_api.py
---------------------------------
POST /plants and GET /plants/{plant_id} exercised end to end (roadmap
P2.3) -- proving src/onboarding/plant_registration.py is wired into
serving, and that a freshly onboarded plant is genuinely usable by
EXISTING endpoints immediately, with no restart and no code change (the
actual substance of the roadmap's own acceptance criterion: "a new plant
is live in under 30 minutes without your involvement").

Every test monkeypatches src.utils.config_loader.PLANTS_DIR to a tmp_path
so nothing ever touches the real configs/plants/ directory.

RUN WITH:
  pytest tests/test_plant_registration_api.py -v
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _valid_payload(**overrides):
    payload = {
        "plant_id": "kolar_75mw",
        "name": "Kolar Solar Plant",
        "location": {
            "name": "Kolar", "state": "Karnataka",
            "latitude": 13.13, "longitude": 78.13, "timezone": "Asia/Kolkata",
        },
        "capacity": {
            "ac_capacity_mw": 75.0, "dc_capacity_mw": 90.0,
            "panel_efficiency": 0.21, "temperature_coefficient": -0.0035,
            "performance_ratio": 0.82, "panel_area_m2": 400000.0,
        },
        "grid": {"export_limit_mw": 70.0},
        "equipment": {
            "commercial_operation_date": "2021-05-01",
            "module_type": "Bifacial 550Wp", "inverter_count": 25,
        },
        "regulatory": {"seller_category": "solar", "contract_rate_rs_per_kwh": 3.0},
    }
    payload.update(overrides)
    return payload


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from src.utils import config_loader
    import src.api.main as main_module

    monkeypatch.setattr(config_loader, "PLANTS_DIR", tmp_path)
    config_loader.get_plant_config.cache_clear()
    yield TestClient(main_module.app)
    config_loader.get_plant_config.cache_clear()


class TestOnboardPlant:
    def test_success_returns_201_and_visible_in_list(self, client):
        resp = client.post("/plants", json=_valid_payload())
        assert resp.status_code == 201
        data = resp.json()
        assert data["plant_id"] == "kolar_75mw"
        assert data["config_path"] == "configs/plants/kolar_75mw.yaml"

        listing = client.get("/plants")
        assert "kolar_75mw" in listing.json()["plant_ids"]

    def test_detail_matches_post_response(self, client):
        resp = client.post("/plants", json=_valid_payload())
        detail = client.get("/plants/kolar_75mw")
        assert detail.status_code == 200
        assert detail.json() == resp.json()["plant_config"]

    def test_onboarded_plant_immediately_usable_by_quality_check(self, client):
        client.post("/plants", json=_valid_payload())
        resp = client.post("/quality/check", json={
            "plant_id": "kolar_75mw",
            "readings": [{"timestamp": "2024-06-01T12:00:00+05:30", "power_mw": 10.0}],
        })
        assert resp.status_code == 200

    def test_onboarded_plant_immediately_usable_by_schedule_risk(self, client):
        client.post("/plants", json=_valid_payload())
        resp = client.post("/schedule/risk", json={
            "plant_id": "kolar_75mw",
            "declared_schedule_mw": [50], "p10_mw": [48], "p50_mw": [50], "p90_mw": [52],
            "as_of": "2024-06-01",
        })
        assert resp.status_code == 200

    def test_path_traversal_plant_id_rejected(self, client, tmp_path):
        resp = client.post("/plants", json=_valid_payload(plant_id="../../etc/passwd"))
        assert resp.status_code == 422
        assert list(tmp_path.iterdir()) == []

    def test_absolute_path_plant_id_rejected(self, client, tmp_path):
        resp = client.post("/plants", json=_valid_payload(plant_id="/etc/passwd"))
        assert resp.status_code == 422
        assert list(tmp_path.iterdir()) == []

    def test_duplicate_returns_409_and_does_not_overwrite(self, client):
        client.post("/plants", json=_valid_payload(name="First"))
        resp = client.post("/plants", json=_valid_payload(name="Second"))
        assert resp.status_code == 409
        detail = client.get("/plants/kolar_75mw")
        assert detail.json()["name"] == "First"

    def test_missing_required_field_returns_422(self, client):
        payload = _valid_payload()
        del payload["capacity"]["performance_ratio"]
        resp = client.post("/plants", json=payload)
        assert resp.status_code == 422

    def test_invalid_seller_category_returns_422(self, client):
        resp = client.post("/plants", json=_valid_payload(
            regulatory={"seller_category": "coal"}
        ))
        assert resp.status_code == 422

    def test_invalid_transaction_type_returns_422(self, client):
        resp = client.post("/plants", json=_valid_payload(
            regulatory={"seller_category": "solar", "transaction_type": "exchange"}
        ))
        assert resp.status_code == 422

    def test_invalid_timezone_returns_422(self, client):
        payload = _valid_payload()
        payload["location"]["timezone"] = "Mars/Phobos"
        resp = client.post("/plants", json=payload)
        assert resp.status_code == 422

    def test_invalid_cod_format_returns_422(self, client):
        payload = _valid_payload()
        payload["equipment"]["commercial_operation_date"] = "15-03-2019"
        resp = client.post("/plants", json=payload)
        assert resp.status_code == 422

    def test_omitting_elevation_and_quality_check_still_works(self, client):
        # Regression test: elevation_m must be OMITTED, not written as
        # null, or pvlib gets None and crashes.
        client.post("/plants", json=_valid_payload())
        resp = client.post("/quality/check", json={
            "plant_id": "kolar_75mw",
            "readings": [{"timestamp": "2024-06-01T12:00:00+05:30", "power_mw": 10.0}],
        })
        assert resp.status_code == 200

    def test_omitting_transaction_type_defaults_to_bilateral_in_gate_closures(self, client):
        # Regression test: transaction_type must be OMITTED, not written
        # as null, or the documented "defaults to bilateral" fallback in
        # /schedule/gate-closures silently breaks.
        client.post("/plants", json=_valid_payload(
            regulatory={"seller_category": "solar"}
        ))
        resp = client.get("/schedule/gate-closures", params={
            "plant_id": "kolar_75mw", "timestamp": "2024-06-01T10:00:00+05:30",
        })
        assert resp.status_code == 200
        assert resp.json()["transaction_type"] == "bilateral"
        assert resp.json()["bilateral_revision"]["allowed"] is True


class TestPlantDetail:
    def test_unknown_id_returns_404(self, client):
        resp = client.get("/plants/does_not_exist")
        assert resp.status_code == 404

    def test_malformed_id_returns_404_not_500(self, client):
        resp = client.get("/plants/..%2F..%2Fetc%2Fpasswd")
        assert resp.status_code == 404
