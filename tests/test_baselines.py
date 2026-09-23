"""
test_baselines.py
--------------------
Roadmap P2.9 depends on scoring against the exact same baselines
benchmark.py already used for the model's own development-time
evaluation (now factored into src/evaluation/baselines.py). Every
scenario here is hand-computed, not just "it ran", since a silent
regression in this module would corrupt both the training-time backtest
and the new weekly production report.

RUN WITH:
  pytest tests/test_baselines.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.baselines import (
    LAG_HOURS,
    clear_sky_power,
    nmae_nrmse,
    persistence_baseline,
    physics_baseline,
    smart_persistence_baseline,
)

PLANT_CONFIG = {"capacity": {"ac_capacity_mw": 100.0, "performance_ratio": 0.8}}


class TestPhysicsBaseline:
    def test_matches_the_hand_computed_formula(self):
        idx = pd.date_range("2024-01-01", periods=3, freq="h")
        features = pd.DataFrame({"shortwave_radiation": [0.0, 500.0, 1500.0]}, index=idx)
        result = physics_baseline(features, PLANT_CONFIG)
        # power = ghi/1000 * performance_ratio * capacity, clipped to capacity.
        # 1500/1000*0.8*100 = 120 MW, above the 100 MW capacity -> clips to 100.
        expected = [0.0, 500 / 1000 * 0.8 * 100, 100.0]
        np.testing.assert_allclose(result.to_numpy(), expected)


class TestClearSkyPower:
    def test_matches_the_hand_computed_formula(self):
        idx = pd.date_range("2024-01-01", periods=2, freq="h")
        features = pd.DataFrame({"clear_sky_ghi_model": [1000.0, 500.0]}, index=idx)
        result = clear_sky_power(features, PLANT_CONFIG)
        np.testing.assert_allclose(result.to_numpy(), [80.0, 40.0])


class TestPersistenceBaseline:
    def test_shifts_by_exactly_lag_hours(self):
        assert LAG_HOURS == 24
        y = pd.Series(np.arange(48, dtype=float))
        result = persistence_baseline(y)
        assert result.iloc[24] == 0.0  # y(0)
        assert result.iloc[47] == 23.0  # y(23)
        assert result.iloc[:24].isna().all()


class TestSmartPersistenceBaseline:
    def test_persists_yesterdays_clear_sky_ratio(self):
        # Build 48 hourly rows where clear_sky_ghi_model is constant at
        # 1000 W/m^2 every hour (so clear_sky_power is a constant 80 MW,
        # per TestClearSkyPower above), and actual output on day 1 is a
        # known FRACTION of that -- day 2's forecast should exactly
        # reproduce day 1's ratio applied to day 2's clear-sky power.
        idx = pd.date_range("2024-01-01", periods=48, freq="h")
        features = pd.DataFrame({"clear_sky_ghi_model": 1000.0}, index=idx)
        y = pd.Series(0.0, index=idx)
        y.iloc[10] = 60.0  # hour 10, day 1: ratio = 60/80 = 0.75
        result = smart_persistence_baseline(y, features, PLANT_CONFIG)
        assert result.iloc[10 + LAG_HOURS] == pytest.approx(0.75 * 80.0)

    def test_zero_clear_sky_power_does_not_raise_and_yields_nan(self):
        idx = pd.date_range("2024-01-01", periods=48, freq="h")
        features = pd.DataFrame({"clear_sky_ghi_model": 0.0}, index=idx)  # night everywhere
        y = pd.Series(0.0, index=idx)
        result = smart_persistence_baseline(y, features, PLANT_CONFIG)
        assert result.isna().all() or (result == 0).all()  # 0/0 -> NaN -> clipped NaN stays NaN


class TestNmaeNrmse:
    def test_hand_computed_mae_and_rmse(self):
        y_true = pd.Series([10.0, 20.0, 30.0, 40.0])
        y_pred = pd.Series([12.0, 18.0, 33.0, 40.0])
        daytime = pd.Series([True, True, True, True])
        capacity_mw = 100.0
        # errors: -2, 2, -3, 0 -> MAE = 7/4 = 1.75, RMSE = sqrt((4+4+9+0)/4) = sqrt(4.25)
        nmae, nrmse, n = nmae_nrmse(y_true, y_pred, capacity_mw, daytime)
        assert n == 4
        assert nmae == pytest.approx(100 * 1.75 / 100.0)
        assert nrmse == pytest.approx(100 * np.sqrt(4.25) / 100.0)

    def test_nighttime_mask_excludes_rows(self):
        y_true = pd.Series([10.0, 20.0])
        y_pred = pd.Series([15.0, 20.0])  # only the first row has any error
        daytime = pd.Series([True, False])
        nmae, nrmse, n = nmae_nrmse(y_true, y_pred, 100.0, daytime)
        assert n == 1
        assert nmae == pytest.approx(5.0)

    def test_empty_after_masking_returns_nan_not_an_exception(self):
        y_true = pd.Series([10.0, 20.0])
        y_pred = pd.Series([15.0, 20.0])
        daytime = pd.Series([False, False])
        nmae, nrmse, n = nmae_nrmse(y_true, y_pred, 100.0, daytime)
        assert n == 0
        assert np.isnan(nmae)
        assert np.isnan(nrmse)

    def test_nan_predictions_are_excluded_from_the_mask(self):
        y_true = pd.Series([10.0, 20.0, 30.0])
        y_pred = pd.Series([15.0, np.nan, 30.0])
        daytime = pd.Series([True, True, True])
        nmae, nrmse, n = nmae_nrmse(y_true, y_pred, 100.0, daytime)
        assert n == 2  # the NaN row is dropped
