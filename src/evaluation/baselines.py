"""
baselines.py — Shared, parameter-free forecast baselines and the scoring
convention used everywhere in this project (originally written for
benchmark.py's rolling-origin backtest and final holdout, factored out
here so roadmap P2.9's weekly production report scores against the
EXACT SAME baselines, not a re-implementation that could quietly drift).

Every score is nMAE / nRMSE as a PERCENTAGE OF THE PLANT'S AC CAPACITY
(never MAPE), against three baselines with zero learned parameters:

  - persistence          forecast(t) = actual(t - 24h)
  - smart_persistence    forecast(t) = clear_sky_power(t) *
                                        [actual(t-24h) / clear_sky_power(t-24h)]
                          (persist YESTERDAY's clear-sky ratio)
  - physics              forecast(t) = forecast_GHI(t) -> power via the
                          plant's rated efficiency only, no learning at all
"""

import numpy as np
import pandas as pd

LAG_HOURS = 24  # how far back persistence/smart-persistence look -- one full day,
                # matching this project's hourly forecast resolution (see README
                # Known Gaps: no per-15-min-block probabilistic forecast yet)

BASELINE_NAMES = ("persistence", "smart_persistence", "physics")


def physics_baseline(features: pd.DataFrame, plant_config: dict) -> pd.Series:
    """No ML: forecast GHI -> power via the plant's rated conversion only."""
    plant = plant_config["capacity"]
    power = (features["shortwave_radiation"] / 1000) * plant["performance_ratio"] * plant["ac_capacity_mw"]
    return power.clip(lower=0, upper=plant["ac_capacity_mw"])


def clear_sky_power(features: pd.DataFrame, plant_config: dict) -> pd.Series:
    plant = plant_config["capacity"]
    return (features["clear_sky_ghi_model"] / 1000) * plant["performance_ratio"] * plant["ac_capacity_mw"]


def persistence_baseline(y: pd.Series) -> pd.Series:
    """D-1 same block: forecast(t) = actual(t - 24h)."""
    return y.shift(LAG_HOURS)


def smart_persistence_baseline(y: pd.Series, features: pd.DataFrame, plant_config: dict) -> pd.Series:
    """Persist YESTERDAY's clear-sky ratio, applied to TODAY's clear-sky
    estimate. Uses only y(t-24h) and clear_sky_power(t-24h)/(t) -- nothing
    at or after t, so this is a legitimate forecast, not a leak."""
    csp = clear_sky_power(features, plant_config)
    ratio_yesterday = (y / csp.replace(0, np.nan)).shift(LAG_HOURS)
    return (csp * ratio_yesterday).clip(lower=0)


def nmae_nrmse(y_true: pd.Series, y_pred: pd.Series, capacity_mw: float,
               daytime_mask: pd.Series):
    """Restricted to daylight hours, for every method equally. Nighttime
    power is trivially ~0 for any method, so including it would flatter
    everyone's score by the same amount EXCEPT smart_persistence, whose
    clear-sky ratio is undefined (0/0) at night and drops out on its own --
    scoring it against a daytime-only mask that every other method also
    uses keeps the comparison apples-to-apples.

    Returns (nmae_pct, nrmse_pct, n) -- nmae_pct/nrmse_pct are NaN (not an
    exception) when n == 0, since "no data to score" is itself meaningful
    information a caller may want to report, not an error."""
    mask = y_true.notna() & y_pred.notna() & daytime_mask
    n = int(mask.sum())
    if n == 0:
        return float("nan"), float("nan"), 0
    err = y_true[mask] - y_pred[mask]
    mae = err.abs().mean()
    rmse = np.sqrt((err ** 2).mean())
    return 100 * mae / capacity_mw, 100 * rmse / capacity_mw, n
