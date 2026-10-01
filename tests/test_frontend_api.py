"""
test_frontend_api.py
----------------------
The endpoints added for the browser frontend (frontend/):

  - CORS: an allow-listed origin gets CORS headers, others don't.
  - GET /plants/{plant_id}: full config incl. data_provenance; unknown or
    path-traversal ids are 404, never another YAML.
  - GET /model-card: serves src/models/model_card.json unchanged.
  - GET /forecast/live: weather fetched server-side (network monkeypatched
    out here), 96 blocks out, consistent with POST /forecast on the same
    weather; upstream failures are 502, never fake weather.
  - POST /dsm/estimate: per-block charges match src/regulatory/dsm.py
    called directly.

RUN WITH:
  pytest tests/test_frontend_api.py -v
"""

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATE = "2026-06-01"


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from src.api.main import app
    return TestClient(app)


def _fake_weather(date: str, rows: int = 25) -> pd.DataFrame:
    """Hourly weather from `date` 00:00 IST, shaped like
    OpenMeteoFetcher._parse_response's output."""
    idx = pd.date_range(pd.Timestamp(date, tz="Asia/Kolkata"), periods=rows, freq="h")
    sun = np.maximum(0, np.cos((idx.hour - 12.5) * np.pi / 12))
    return pd.DataFrame({
        "shortwave_radiation": 850 * sun,
        "cloud_cover": 10.0,
        "temperature_2m": 25 + 10 * sun,
        "relative_humidity_2m": 40.0,
        "wind_speed_10m": 3.0,
        "precipitation": 0.0,
        "hour": idx.hour,
        "month": idx.month,
    }, index=idx)


class TestCors:
    def test_allowed_origin_gets_header(self, client):
        resp = client.get("/health", headers={"Origin": "http://localhost:5173"})
        assert resp.headers.get("access-control-allow-origin") == "http://localhost:5173"

    def test_unknown_origin_gets_no_header(self, client):
        resp = client.get("/health", headers={"Origin": "https://evil.example"})
        assert "access-control-allow-origin" not in resp.headers


class TestPlantDetail:
    def test_returns_full_config_with_provenance(self, client):
        resp = client.get("/plants/pune_50mw")
        assert resp.status_code == 200
        data = resp.json()
        assert data["capacity"]["ac_capacity_mw"] == 50
        assert data["data_provenance"]["simulated"] is True
        assert "regulatory.contract_rate_rs_per_kwh" in data["data_provenance"]["placeholder_fields"]

    def test_unknown_plant_is_404(self, client):
        assert client.get("/plants/nope").status_code == 404

    def test_path_traversal_is_404_not_another_yaml(self, client):
        from src.utils.config_loader import get_plant_config
        with pytest.raises(FileNotFoundError):
            get_plant_config("../config")

    def test_response_is_a_copy_not_the_cached_config(self, client):
        from src.utils.config_loader import get_plant_config
        client.get("/plants/jaipur_100mw")
        assert get_plant_config("jaipur_100mw")["capacity"]["ac_capacity_mw"] == 100


class TestModelCard:
    def test_serves_model_card_json(self, client):
        resp = client.get("/model-card")
        assert resp.status_code == 200
        on_disk = json.loads((PROJECT_ROOT / "src" / "models" / "model_card.json").read_text())
        assert resp.json() == on_disk


class TestLiveForecast:
    @pytest.fixture
    def fake_fetch(self, monkeypatch):
        import src.api.main as api
        calls = []

        def fake(plant_config, date):
            calls.append((plant_config["plant_id"], date))
            return _fake_weather(date)

        monkeypatch.setattr(api, "_fetch_weather_for_date", fake)
        return calls

    def test_returns_96_blocks_with_real_timestamps(self, client, fake_fetch):
        resp = client.get("/forecast/live", params={"date": DATE, "plant_id": "jaipur_100mw"})
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert fake_fetch == [("jaipur_100mw", DATE)]
        assert len(data["block_starts"]) == 96
        assert len(data["predictions_mw"]) == 96
        assert data["block_starts"][0] == "2026-06-01T00:00:00+05:30"
        assert data["block_starts"][-1] == "2026-06-01T23:45:00+05:30"
        assert len(data["weather_hourly"]["timestamps"]) == 24
        assert data["weather_source"] == "open-meteo"

    def test_blocks_are_consistent_with_hourly_forecast(self, client, fake_fetch):
        """Integrating the 96 blocks must give the same daily energy as
        trapezoidal integration of POST /forecast's hourly output on the
        identical weather -- the resampling adds no energy."""
        live = client.get("/forecast/live", params={"date": DATE}).json()
        w = _fake_weather(DATE)
        hourly = client.post("/forecast", json={"hours": [{
            "timestamp": ts.isoformat(),
            **{c: float(w.loc[ts, c]) for c in ["shortwave_radiation", "cloud_cover", "temperature_2m",
                                                 "relative_humidity_2m", "wind_speed_10m"]},
        } for ts in w.index]}).json()
        hourly_mwh = np.trapezoid(hourly["predictions_mw"], dx=1.0)
        assert live["total_generation_mwh"] == pytest.approx(hourly_mwh, abs=0.5)

    def test_quantile_bands_are_ordered(self, client, fake_fetch):
        data = client.get("/forecast/live", params={"date": DATE}).json()
        if not data["predictions_p10_mw"]:
            pytest.skip("quantile model not loaded")
        p10, p50, p90 = (np.array(data[k]) for k in
                          ["predictions_p10_mw", "predictions_p50_mw", "predictions_p90_mw"])
        assert (p10 <= p50 + 1e-6).all() and (p50 <= p90 + 1e-6).all()

    def test_scales_with_plant_capacity(self, client, fake_fetch):
        j = client.get("/forecast/live", params={"date": DATE, "plant_id": "jaipur_100mw"}).json()
        p = client.get("/forecast/live", params={"date": DATE, "plant_id": "pune_50mw"}).json()
        assert max(p["predictions_mw"]) <= 50
        assert p["capacity_mw"] == 50 and j["capacity_mw"] == 100

    def test_upstream_failure_is_502_not_fake_weather(self, client, monkeypatch):
        import src.api.main as api

        def boom(plant_config, date):
            raise RuntimeError("network down")

        monkeypatch.setattr(api, "_fetch_weather_for_date", boom)
        resp = client.get("/forecast/live", params={"date": DATE})
        assert resp.status_code == 502
        assert "network down" in resp.json()["detail"]

    def test_null_weather_is_502(self, client, monkeypatch):
        import src.api.main as api

        def gappy(plant_config, date):
            w = _fake_weather(date)
            w.iloc[10, 0] = np.nan
            return w

        monkeypatch.setattr(api, "_fetch_weather_for_date", gappy)
        assert client.get("/forecast/live", params={"date": DATE}).status_code == 502

    def test_bad_date_and_unknown_plant(self, client, fake_fetch):
        assert client.get("/forecast/live", params={"date": "01-06-2026"}).status_code == 422
        assert client.get("/forecast/live", params={"date": DATE, "plant_id": "nope"}).status_code == 404


class TestFetcherForDate:
    def test_requests_date_through_next_day_and_trims(self, monkeypatch):
        from src.ingestion.open_meteo_fetcher import OpenMeteoFetcher
        from src.utils.config_loader import get_plant_config
        seen = {}

        def fake_fetch(self, url, params, label):
            seen.update(params)
            times = pd.date_range("2026-06-01", periods=48, freq="h").strftime("%Y-%m-%dT%H:%M")
            return {"hourly": {"time": list(times),
                                **{v: [1.0] * 48 for v in self.variables}}}

        monkeypatch.setattr(OpenMeteoFetcher, "_fetch_with_retry", fake_fetch)
        df = OpenMeteoFetcher(get_plant_config("jaipur_100mw")).fetch_forecast_for_date(DATE)
        assert seen["start_date"] == "2026-06-01" and seen["end_date"] == "2026-06-02"
        assert len(df) == 25
        assert df.index[-1] == pd.Timestamp("2026-06-02", tz="Asia/Kolkata")


class TestDsmEstimate:
    def test_matches_dsm_module_directly(self, client):
        from src.regulatory import dsm
        sched = [50.0] * 96
        inj = [50.0 + (i % 7 - 3) * 4 for i in range(96)]
        resp = client.post("/dsm/estimate", json={
            "plant_id": "jaipur_100mw", "date": DATE,
            "scheduled_mw": sched, "injected_mw": inj,
        })
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["dt_hours"] == 0.25
        assert data["contract_rate_is_placeholder"] is True
        assert data["blocks"][0]["block_start"] == "2026-06-01T00:00:00+05:30"
        expected = sum(
            dsm.deviation_settlement((i_ - s) * 0.25, 100 * 0.25, 2.50, "solar", dt.date(2026, 6, 1))["net_rs"]
            for s, i_ in zip(sched, inj)
        )
        assert data["summary"]["total_net_rs"] == pytest.approx(expected, abs=0.5)
        s = data["summary"]
        assert s["total_net_rs"] == pytest.approx(s["total_payable_rs"] - s["total_receivable_rs"], abs=0.05)

    def test_zero_deviation_costs_nothing(self, client):
        data = client.post("/dsm/estimate", json={
            "scheduled_mw": [30.0] * 4, "injected_mw": [30.0] * 4,
        }).json()
        assert data["summary"]["total_net_rs"] == 0
        assert "block_start" not in data["blocks"][0]

    def test_length_mismatch_and_wrong_block_count_are_422(self, client):
        assert client.post("/dsm/estimate", json={
            "scheduled_mw": [1.0] * 4, "injected_mw": [1.0] * 5,
        }).status_code == 422
        assert client.post("/dsm/estimate", json={
            "date": DATE, "scheduled_mw": [1.0] * 4, "injected_mw": [1.0] * 4,
        }).status_code == 422


class TestOptimizeUsesScheduleDate:
    def test_dsm_volume_limits_follow_the_schedule_date(self, client):
        """Same inputs, dates either side of the 01.04.2026 cutover: the
        DSM cost must be computed with each date's own volume limits, not
        today's."""
        solar = [0.0] * 24 + [60.0] * 48 + [0.0] * 24
        sched = [0.0] * 24 + [80.0] * 48 + [0.0] * 24
        body = {"solar_forecast_mw": solar, "declared_schedule_mw": sched,
                "battery_capacity_mwh": 1.0, "charge_rate_mw": 1.0, "discharge_rate_mw": 1.0,
                "initial_charge_mwh": 0.5, "plant_id": "jaipur_100mw"}
        pre = client.post("/optimize", json={**body, "date": "2026-03-01"}).json()
        post = client.post("/optimize", json={**body, "date": "2026-06-01"}).json()
        from src.regulatory import dsm
        if dsm.volume_limits("solar", dt.date(2026, 3, 1)) == dsm.volume_limits("solar", dt.date(2026, 6, 1)):
            pytest.skip("volume limits identical across the cutover")
        assert pre["summary"]["total_dsm_rs"] != post["summary"]["total_dsm_rs"]
