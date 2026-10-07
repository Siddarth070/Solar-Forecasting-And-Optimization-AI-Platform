"""
test_weekly_report.py
------------------------
Roadmap P2.9 acceptance criterion: "contains no number that cannot be
traced to raw data." Builds a small, fully hand-computable
forecast-vs-actual dataset (constant clear-sky index of 1.0, so the
physics baseline is a known constant; a day-1 constant actual and a
different day-2 constant actual, so persistence/smart_persistence carry
an exact, known error) and checks every reported nMAE against arithmetic
worked out by hand, not just "it ran".

RUN WITH:
  pytest tests/test_weekly_report.py -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.reporting.weekly_report import generate_weekly_report
from src.utils.config_loader import get_plant_config

PLANT = get_plant_config("jaipur_100mw")
CAPACITY_MW = PLANT["capacity"]["ac_capacity_mw"]  # 100
PERFORMANCE_RATIO = PLANT["capacity"]["performance_ratio"]  # 0.8


def _features_and_forecasts(report_predicted_mw):
    """30 hourly rows: hours 0-23 (day 1) actual=50 MW constant, hours
    24-29 (day 2, the 'report window') actual=55 MW constant, clear-sky
    index = 1.0 everywhere (shortwave_radiation == clear_sky_ghi_model =
    1000 W/m^2) so physics_baseline is the known constant
    capacity*performance_ratio = 80 MW at every hour."""
    idx = pd.date_range("2024-01-01 00:00", periods=30, freq="h", tz="Asia/Kolkata")
    features = pd.DataFrame({
        "shortwave_radiation": 1000.0,
        "clear_sky_ghi_model": 1000.0,
        "solar_output_mw": [50.0] * 24 + [55.0] * 6,
    }, index=idx)

    report_idx = idx[24:30]
    forecasts = pd.DataFrame({
        "target_timestamp": report_idx,
        "horizon_hours": [1, 1, 1, 6, 6, 6],
        "predicted_mw": report_predicted_mw,
    })
    return features, forecasts


class TestOverallAndBaselineScoring:
    def test_hand_computed_nmae_for_model_and_all_three_baselines(self):
        # predicted_mw errors vs actual=55: 2, 1, 1, 5, 5, 0 -> mean 14/6
        features, forecasts = _features_and_forecasts([53.0, 54.0, 56.0, 50.0, 60.0, 55.0])
        report = generate_weekly_report(forecasts, features, PLANT)

        assert report.n_forecasts == 6
        # the module rounds to 3 decimals, so compare with matching tolerance
        assert report.overall["model"]["nmae_pct"] == pytest.approx(100 * (14 / 6) / CAPACITY_MW, abs=1e-3)

        # persistence(t) = actual(t-24h) = 50 MW (day 1) for every report
        # row -> error = 55-50 = 5 MW constant -> nMAE = 5%.
        assert report.overall["persistence"]["nmae_pct"] == pytest.approx(5.0)
        # smart_persistence: clear-sky power is a constant 80 MW, so
        # yesterday's ratio (50/80) applied to today's clear-sky power
        # reproduces the same 50 MW prediction -> identical to persistence here.
        assert report.overall["smart_persistence"]["nmae_pct"] == pytest.approx(5.0)
        # physics = ghi/1000 * performance_ratio * capacity = 80 MW constant
        # -> error = 55-80 = -25 -> nMAE = 25%.
        assert report.overall["physics"]["nmae_pct"] == pytest.approx(25.0)

    def test_model_beating_every_baseline_is_reported_true(self):
        features, forecasts = _features_and_forecasts([53.0, 54.0, 56.0, 50.0, 60.0, 55.0])
        report = generate_weekly_report(forecasts, features, PLANT)
        assert report.beats_baseline == {
            "persistence": True, "smart_persistence": True, "physics": True,
        }

    def test_model_losing_to_persistence_is_reported_false(self):
        # predicted_mw = 0 for every report row -> error = 55 MW constant
        # -> model nMAE 55% > persistence's 5% -> model does NOT beat it.
        features, forecasts = _features_and_forecasts([0.0] * 6)
        report = generate_weekly_report(forecasts, features, PLANT)
        assert report.overall["model"]["nmae_pct"] == pytest.approx(55.0)
        assert report.beats_baseline["persistence"] is False
        assert report.beats_baseline["smart_persistence"] is False
        # physics (25%) is still worse than the model's 55%? No -- 55 > 25,
        # so the model does NOT beat physics either here.
        assert report.beats_baseline["physics"] is False


class TestByHorizonAndByBlock:
    def test_by_horizon_splits_errors_into_the_right_buckets(self):
        # horizon=1 rows (errors 2,1,1) -> MAE 4/3; horizon=6 rows
        # (errors 5,5,0) -> MAE 10/3.
        features, forecasts = _features_and_forecasts([53.0, 54.0, 56.0, 50.0, 60.0, 55.0])
        report = generate_weekly_report(forecasts, features, PLANT)

        assert set(report.by_horizon.keys()) == {1, 6}
        assert report.by_horizon[1]["model"]["n"] == 3
        assert report.by_horizon[1]["model"]["nmae_pct"] == pytest.approx(100 * (4 / 3) / CAPACITY_MW, abs=1e-3)
        assert report.by_horizon[6]["model"]["n"] == 3
        assert report.by_horizon[6]["model"]["nmae_pct"] == pytest.approx(100 * (10 / 3) / CAPACITY_MW, abs=1e-3)

    def test_by_block_splits_into_six_distinct_hours_of_day(self):
        features, forecasts = _features_and_forecasts([53.0, 54.0, 56.0, 50.0, 60.0, 55.0])
        report = generate_weekly_report(forecasts, features, PLANT)
        # target hours 24-29 land on calendar hours 0,1,2,3,4,5 of day 2.
        assert set(report.by_block.keys()) == {0, 1, 2, 3, 4, 5}
        # hour 0 (index 24, predicted 53, actual 55) -> single-row bucket,
        # error 2 MW -> nMAE exactly 2%.
        assert report.by_block[0]["model"]["n"] == 1
        assert report.by_block[0]["model"]["nmae_pct"] == pytest.approx(2.0)


class TestValidation:
    def test_empty_forecasts_raises(self):
        features, forecasts = _features_and_forecasts([53.0, 54.0, 56.0, 50.0, 60.0, 55.0])
        with pytest.raises(ValueError, match="empty"):
            generate_weekly_report(forecasts.iloc[0:0], features, PLANT)

    def test_target_timestamp_missing_from_features_raises(self):
        features, forecasts = _features_and_forecasts([53.0, 54.0, 56.0, 50.0, 60.0, 55.0])
        bad_forecasts = forecasts.copy()
        bad_forecasts.loc[0, "target_timestamp"] = pd.Timestamp("2030-01-01T00:00:00+05:30")
        with pytest.raises(ValueError, match="not covered"):
            generate_weekly_report(bad_forecasts, features, PLANT)
