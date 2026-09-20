"""
pipeline.py — Feature engineering for the solar forecasting model.
--------------------------------------------------------------------

RULE: no feature computed here may read `solar_output_mw` (the
forecasting target) at or after the timestamp it is computed for.

Historically this pipeline lived only in `notebooks/feature_engineering.ipynb`,
and its `clear_sky_ratio` feature was

    clear_sky_ratio = solar_output_mw / clear_sky_output

— the target divided by a same-timestamp feature. That is target leakage:
the ratio is only computable once you already know the answer you're
trying to predict, so it is useless (and dishonestly flattering) for a
real forecast.

The fix (`clear_sky_index` below) replaces the leaky denominator's partner
with a value that is knowable ahead of time: forecast/observed GHI
(`shortwave_radiation`, a weather variable) divided by a clear-sky GHI
estimate from a solar-position model (pvlib's Ineichen model), which
depends only on location and timestamp — never on any generation value.
"""

import numpy as np
import pandas as pd
import pvlib

# Columns the trained model expects, in order. Any change here is a
# breaking change to the served model and must ship with a retrained
# artifact (see P0.3 — feature/serving parity).
FEATURE_COLUMNS = [
    "cloud_cover",
    "shortwave_radiation",
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "solar_lag_1h",
    "solar_lag_24h",
    "solar_lag_48h",
    "solar_lag_168h",
    "solar_rolling_mean_3h",
    "solar_rolling_mean_6h",
    "solar_rolling_std_3h",
    "clear_sky_index",
]


def clear_sky_ghi(index: pd.DatetimeIndex, latitude: float, longitude: float,
                   altitude: float = 0.0) -> pd.Series:
    """
    Physics-based clear-sky GHI (W/m^2) from pvlib's Ineichen model.

    Depends only on solar position for `index` at (`latitude`, `longitude`,
    `altitude`) — never on any measured or simulated generation value, so
    it is computable for a timestamp that hasn't happened yet.
    """
    location = pvlib.location.Location(latitude, longitude, altitude=altitude)
    return location.get_clearsky(index, model="ineichen")["ghi"]


def build_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Build the model's full feature set from a raw weather (+ optional
    generation) frame.

    Parameters
    ----------
    df : pd.DataFrame
        Indexed by a tz-aware timestamp. Must contain `hour`, `month`,
        `shortwave_radiation`, `cloud_cover`, `temperature_2m`,
        `relative_humidity_2m`, `wind_speed_10m`. May optionally contain
        `solar_output_mw` (only present for historical/training rows —
        a genuine forecast row has no generation value yet).
    config : dict
        Loaded app config (see `src.utils.config_loader.get_config`).
        Reads `config["location"]["latitude"/"longitude"/"elevation_m"]`.

    Returns
    -------
    pd.DataFrame
        `df` plus every engineered feature column.
    """
    latitude = config["location"]["latitude"]
    longitude = config["location"]["longitude"]
    altitude = config["location"].get("elevation_m", 0.0)

    out = df.copy()

    # ── Cyclical time encodings — depend only on the calendar ───────────
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)

    # ── Clear-sky index — leak-free replacement for clear_sky_ratio ─────
    # Numerator: shortwave_radiation, a weather variable available from a
    # forecast ahead of time. Denominator: a solar-position-only estimate,
    # computable for any future timestamp with zero generation data.
    csghi = clear_sky_ghi(out.index, latitude, longitude, altitude).to_numpy()
    out["clear_sky_ghi_model"] = csghi
    with np.errstate(divide="ignore", invalid="ignore"):
        raw_index = out["shortwave_radiation"].to_numpy() / csghi
    out["clear_sky_index"] = np.where(csghi > 1.0, raw_index, 0.0)
    # Cloud enhancement can briefly push observed GHI above the modelled
    # clear-sky value, so allow a little headroom above 1.0 rather than
    # clipping real signal away.
    out["clear_sky_index"] = out["clear_sky_index"].clip(0, 1.3)

    # ── Autoregressive features — legitimate use of PAST target values ──
    # Only present when solar_output_mw (history) is available at all, and
    # every one is shifted so that the value for row t never includes row
    # t's own target — only t-1 and earlier.
    if "solar_output_mw" in out.columns:
        past_target = out["solar_output_mw"].shift(1)
        out["solar_lag_1h"] = out["solar_output_mw"].shift(1)
        out["solar_lag_24h"] = out["solar_output_mw"].shift(24)
        out["solar_lag_48h"] = out["solar_output_mw"].shift(48)
        out["solar_lag_168h"] = out["solar_output_mw"].shift(168)
        out["solar_rolling_mean_3h"] = past_target.rolling(3).mean()
        out["solar_rolling_mean_6h"] = past_target.rolling(6).mean()
        out["solar_rolling_std_3h"] = past_target.rolling(3).std()

    return out
