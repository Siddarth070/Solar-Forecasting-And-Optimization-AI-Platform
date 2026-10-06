"""
test_web_api_support.py
-----------------------
The small API surface the web console (web/) depends on: plant detail,
model card, CORS, and energy totals that are correct at 15-minute
resolution (a 96-block request used to report 4x the true MWh).
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


def _row(ts, ghi):
    return {"timestamp": ts, "shortwave_radiation": ghi, "cloud_cover": 10,
            "temperature_2m": 30, "relative_humidity_2m": 40, "wind_speed_10m": 3}


def test_plant_detail(client):
    r = client.get("/plants/jaipur_100mw")
    assert r.status_code == 200
    body = r.json()
    assert body["ac_capacity_mw"] == 100
    assert body["location"]["timezone"] == "Asia/Kolkata"
    assert body["regulatory"]["seller_category"] == "solar"
    assert body["simulated"] is True


def test_plant_detail_unknown(client):
    assert client.get("/plants/does_not_exist").status_code == 404


def test_model_card(client):
    r = client.get("/model-card")
    assert r.status_code == 200
    assert "rolling_origin_backtest" in r.json()


def test_energy_is_power_times_interval_at_15_min(client):
    ts = [f"2026-06-01T{h:02d}:{m:02d}:00+05:30" for h in range(10, 12) for m in (0, 15, 30, 45)]
    r = client.post("/forecast", json={"hours": [_row(t, 800) for t in ts]})
    assert r.status_code == 200
    body = r.json()
    assert body["interval_hours"] == pytest.approx(0.25)
    assert body["total_generation_mwh"] == pytest.approx(sum(body["predictions_mw"]) * 0.25, abs=0.05)


def test_energy_hourly_unchanged(client):
    ts = [f"2026-06-01T{h:02d}:00:00+05:30" for h in range(10, 13)]
    body = client.post("/forecast", json={"hours": [_row(t, 800) for t in ts]}).json()
    assert body["interval_hours"] == pytest.approx(1.0)
    assert body["total_generation_mwh"] == pytest.approx(sum(body["predictions_mw"]), abs=0.05)


def test_cors_allows_local_console(client):
    r = client.options("/health", headers={"Origin": "http://localhost:5173",
                                            "Access-Control-Request-Method": "GET"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_optimize_uses_schedule_date_for_dsm_cutover(client):
    # 5 MW over-delivery in every block of a 100 MW plant = 1.25 MWh/block,
    # exactly VL1 post-cutover (5% of 25 MWh) but well inside VL1 pre-cutover
    # (10%). Pre-cutover there's no band-2 haircut; post-cutover there's none
    # either at exactly the edge, so use 7 MW (1.75 MWh) to straddle.
    body = {"solar_forecast_mw": [57.0] * 96, "declared_schedule_mw": [50.0] * 96,
            "battery_capacity_mwh": 0, "charge_rate_mw": 0, "discharge_rate_mw": 0,
            "initial_charge_mwh": 0, "plant_id": "jaipur_100mw"}
    pre = client.post("/optimize", json={**body, "date": "2026-03-15"}).json()
    post = client.post("/optimize", json={**body, "date": "2026-06-15"}).json()
    # Pre: all 1.75 MWh paid at 1.0x. Post: 1.25 at 1.0x + 0.5 at 0.9x.
    assert pre["summary"]["total_dsm_rs"] == pytest.approx(-96 * 1.75 * 2500, rel=1e-3)
    assert post["summary"]["total_dsm_rs"] == pytest.approx(-96 * (1.25 + 0.45) * 2500, rel=1e-3)
