"""
test_no_leakage.py
-------------------
Guards against target leakage in src/features/pipeline.py.

THE RULE:
  No feature's computation may read `solar_output_mw` (the forecasting
  target) at or after the timestamp it is computed for. Reading PAST
  target values (t-1 and earlier) as autoregressive features is fine.

HOW WE TEST IT:
  Perturb `solar_output_mw` at exactly one timestamp t0, rebuild the
  features, and assert nothing changed at t0 or at any timestamp before
  it. A feature at row t0 that reads the target "at or after" t0 would
  change under this perturbation; a legitimate lag/rolling feature at
  t0 only reads target values strictly before t0, so it is unaffected.
  Rows after t0 are allowed to change — that's exactly what a correct
  autoregressive feature (e.g. solar_lag_24h) is supposed to do.

RUN WITH:
  pytest tests/test_no_leakage.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.pipeline import build_features, FEATURE_COLUMNS
from src.utils.config_loader import get_plant_config


def _sample_frame(n_hours: int = 200) -> pd.DataFrame:
    """Small, deterministic weather+generation frame for testing."""
    index = pd.date_range("2024-06-01", periods=n_hours, freq="h", tz="Asia/Kolkata")
    hours = index.hour.to_numpy()
    months = index.month.to_numpy()
    rng = np.random.default_rng(0)

    solar_factor = np.clip(np.cos((hours - 12.5) * np.pi / 12), 0, None)
    shortwave = 800 * solar_factor
    solar_output = np.clip(80 * solar_factor + rng.normal(0, 0.1, n_hours), 0, 100)

    return pd.DataFrame(
        {
            "hour": hours,
            "month": months,
            "shortwave_radiation": shortwave,
            "cloud_cover": rng.uniform(0, 100, n_hours),
            "temperature_2m": rng.uniform(15, 40, n_hours),
            "relative_humidity_2m": rng.uniform(20, 80, n_hours),
            "wind_speed_10m": rng.uniform(0, 10, n_hours),
            "solar_output_mw": solar_output,
        },
        index=index,
    )


class TestNoLeakage:
    """No feature may read solar_output_mw at or after its own timestamp."""

    def test_perturbing_target_at_t0_does_not_change_features_at_or_before_t0(self):
        plant_config = get_plant_config("jaipur_100mw")
        df = _sample_frame()
        t0 = 100

        baseline = build_features(df, plant_config)

        perturbed_df = df.copy()
        perturbed_df.iloc[
            t0, perturbed_df.columns.get_loc("solar_output_mw")
        ] += 1.0
        perturbed = build_features(perturbed_df, plant_config)

        for col in FEATURE_COLUMNS:
            assert col in baseline.columns, f"missing feature column: {col}"
            pd.testing.assert_series_equal(
                baseline[col].iloc[: t0 + 1],
                perturbed[col].iloc[: t0 + 1],
                check_names=False,
                obj=col,
            )

    def test_features_after_t0_are_free_to_use_the_perturbed_target(self):
        """Sanity check on the test above: autoregressive features SHOULD
        change for rows after t0, proving the equality above isn't trivially
        true because build_features ignores solar_output_mw altogether."""
        plant_config = get_plant_config("jaipur_100mw")
        df = _sample_frame()
        t0 = 100

        baseline = build_features(df, plant_config)

        perturbed_df = df.copy()
        perturbed_df.iloc[
            t0, perturbed_df.columns.get_loc("solar_output_mw")
        ] += 50.0
        perturbed = build_features(perturbed_df, plant_config)

        # solar_lag_1h at t0+1 reads solar_output_mw[t0] — it must move.
        assert not np.isclose(
            baseline["solar_lag_1h"].iloc[t0 + 1],
            perturbed["solar_lag_1h"].iloc[t0 + 1],
        )

    def test_clear_sky_index_is_forward_computable(self):
        """The clear-sky denominator must depend only on time and location
        — identical whether or not solar_output_mw is present at all, as
        it would be absent for a genuine future forecast row."""
        plant_config = get_plant_config("jaipur_100mw")
        df = _sample_frame()
        df_no_target = df.drop(columns=["solar_output_mw"])

        with_target = build_features(df, plant_config)
        without_target = build_features(df_no_target, plant_config)

        pd.testing.assert_series_equal(
            with_target["clear_sky_index"],
            without_target["clear_sky_index"],
            check_names=False,
        )
