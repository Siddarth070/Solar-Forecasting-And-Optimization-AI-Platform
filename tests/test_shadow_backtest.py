"""
test_shadow_backtest.py
------------------------
Unit tests for src/evaluation/shadow_backtest.py (roadmap P3.2).

IMPORTANT: every "actual readings" series in this file is a SYNTHETIC
fixture built for the test, not real plant data -- P3.1 (securing a
real plant's history) is not done yet. Nothing here is presented as a
real P3.2 result; these tests only prove the harness itself works
end-to-end, so it's ready the moment real data lands. The Open-Meteo
fetcher is always mocked/injected -- no test hits the live network.

RUN WITH:
  pytest tests/test_shadow_backtest.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.shadow_backtest import (
    ShadowBacktestError,
    load_actual_readings_csv,
    run_shadow_backtest,
    shadow_backtest_from_csv,
)
from src.utils.config_loader import get_plant_config


def _write_csv(tmp_path, rows):
    path = tmp_path / "actuals.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


class TestLoadActualReadingsCsv:
    def test_valid_csv_parses(self, tmp_path):
        path = _write_csv(tmp_path, [
            {"timestamp": "2024-06-01T06:00:00+05:30", "power_mw": 0.0},
            {"timestamp": "2024-06-01T07:00:00+05:30", "power_mw": 12.5},
        ])
        series = load_actual_readings_csv(path)
        assert isinstance(series.index, pd.DatetimeIndex)
        assert list(series.to_numpy()) == [0.0, 12.5]

    def test_missing_column_raises(self, tmp_path):
        path = _write_csv(tmp_path, [{"timestamp": "2024-06-01T06:00:00+05:30", "mw": 1.0}])
        with pytest.raises(ShadowBacktestError, match="missing required column"):
            load_actual_readings_csv(path)

    def test_mixed_tz_aware_and_naive_raises(self, tmp_path):
        path = _write_csv(tmp_path, [
            {"timestamp": "2024-06-01T06:00:00+05:30", "power_mw": 1.0},
            {"timestamp": "2024-06-01T07:00:00", "power_mw": 2.0},
        ])
        with pytest.raises(ShadowBacktestError, match="one consistent timezone"):
            load_actual_readings_csv(path)

    def test_non_numeric_power_raises(self, tmp_path):
        path = _write_csv(tmp_path, [{"timestamp": "2024-06-01T06:00:00+05:30", "power_mw": "not-a-number"}])
        with pytest.raises(ShadowBacktestError, match="numeric"):
            load_actual_readings_csv(path)

    def test_duplicate_timestamps_deduplicated_keeping_first(self, tmp_path):
        path = _write_csv(tmp_path, [
            {"timestamp": "2024-06-01T06:00:00+05:30", "power_mw": 1.0},
            {"timestamp": "2024-06-01T06:00:00+05:30", "power_mw": 999.0},
        ])
        series = load_actual_readings_csv(path)
        assert len(series) == 1
        assert series.iloc[0] == 1.0


def _fake_weather(index: pd.DatetimeIndex) -> pd.DataFrame:
    """A synthetic weather fixture shaped like OpenMeteoFetcher's real
    output -- a daytime bell curve for shortwave_radiation, flat values
    for everything else. Not real weather; only used to prove the
    plumbing (fetch -> build_features -> score) works."""
    hours = index.hour.to_numpy()
    shortwave = np.clip(800 * np.sin(np.pi * (hours - 6) / 12), 0, None)
    return pd.DataFrame({
        "shortwave_radiation": shortwave,
        "cloud_cover": np.full(len(index), 20.0),
        "temperature_2m": np.full(len(index), 30.0),
        "relative_humidity_2m": np.full(len(index), 40.0),
        "wind_speed_10m": np.full(len(index), 3.0),
        "hour": hours,
        "month": index.month.to_numpy(),
    }, index=index)


@pytest.fixture()
def jaipur_config():
    return get_plant_config("jaipur_100mw")


class TestRunShadowBacktest:
    def test_baselines_only_no_served_model(self, jaipur_config):
        index = pd.date_range("2024-06-01", periods=48, freq="h", tz="Asia/Kolkata")
        weather = _fake_weather(index)
        capacity = jaipur_config["capacity"]["ac_capacity_mw"]
        hours = index.hour.to_numpy()
        actual = pd.Series(
            np.clip(0.6 * capacity * np.sin(np.pi * (hours - 6) / 12), 0, capacity),
            index=index, name="power_mw",
        )

        fake_fetcher = MagicMock()
        fake_fetcher.fetch_historical.return_value = weather

        result = run_shadow_backtest(actual, jaipur_config, meteo_fetcher=fake_fetcher)

        assert result["plant_id"] == "jaipur_100mw"
        assert result["n_readings"] == 48
        assert set(result["scores"].keys()) == {"persistence", "smart_persistence", "physics"}
        for name, s in result["scores"].items():
            assert s["n"] >= 0
            if s["n"] > 0:
                assert 0 <= s["nmae_pct"] < 1000  # sane range, not a fabricated exact number
        fake_fetcher.fetch_historical.assert_called_once_with("2024-06-01", "2024-06-02")

    def test_served_model_scored_when_provided(self, jaipur_config):
        index = pd.date_range("2024-06-01", periods=48, freq="h", tz="Asia/Kolkata")
        weather = _fake_weather(index)
        capacity = jaipur_config["capacity"]["ac_capacity_mw"]
        hours = index.hour.to_numpy()
        actual = pd.Series(
            np.clip(0.6 * capacity * np.sin(np.pi * (hours - 6) / 12), 0, capacity),
            index=index, name="power_mw",
        )

        fake_fetcher = MagicMock()
        fake_fetcher.fetch_historical.return_value = weather

        fake_model = MagicMock()
        fake_model.predict.return_value = np.full(len(index), 0.5)  # capacity fraction

        result = run_shadow_backtest(actual, jaipur_config, served_model=fake_model, meteo_fetcher=fake_fetcher)

        assert "model" in result["scores"]
        assert fake_model.predict.called

    def test_empty_actuals_raises(self, jaipur_config):
        with pytest.raises(ShadowBacktestError, match="empty"):
            run_shadow_backtest(pd.Series(dtype=float), jaipur_config, meteo_fetcher=MagicMock())


class TestShadowBacktestFromCsv:
    def test_bad_quality_data_blocks_before_any_network_call(self, tmp_path, jaipur_config):
        capacity = jaipur_config["capacity"]["ac_capacity_mw"]
        index = pd.date_range("2024-06-01T06:00:00+05:30", periods=10, freq="h")
        path = tmp_path / "actuals.csv"
        pd.DataFrame({
            "timestamp": index.astype(str),
            # Wildly above rated capacity -- a real, obviously-bad export.
            "power_mw": [capacity * 5] * 10,
        }).to_csv(path, index=False)

        fake_fetcher = MagicMock()
        result = shadow_backtest_from_csv(path, "jaipur_100mw", meteo_fetcher=fake_fetcher)

        assert result["status"] == "blocked_on_data_quality"
        assert result["quality_report"]["passed"] is False
        fake_fetcher.fetch_historical.assert_not_called()

    def test_clean_data_runs_end_to_end(self, tmp_path, jaipur_config):
        capacity = jaipur_config["capacity"]["ac_capacity_mw"]
        index = pd.date_range("2024-06-01T00:00:00+05:30", periods=48, freq="h")
        hours = index.hour.to_numpy()
        power = np.clip(0.6 * capacity * np.sin(np.pi * (hours - 6) / 12), 0, capacity)
        path = tmp_path / "actuals.csv"
        pd.DataFrame({"timestamp": index.astype(str), "power_mw": power}).to_csv(path, index=False)

        fake_fetcher = MagicMock()
        fake_fetcher.fetch_historical.return_value = _fake_weather(
            pd.DatetimeIndex(index).tz_convert("Asia/Kolkata")
            if pd.DatetimeIndex(index).tz is not None
            else pd.DatetimeIndex(index).tz_localize("Asia/Kolkata")
        )

        result = shadow_backtest_from_csv(path, "jaipur_100mw", meteo_fetcher=fake_fetcher)

        assert result["status"] == "ok"
        assert result["quality_report"]["passed"] is True
        assert set(result["scores"].keys()) == {"persistence", "smart_persistence", "physics"}
