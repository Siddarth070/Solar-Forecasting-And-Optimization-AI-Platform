"""
benchmark.py — Honest evaluation harness for the solar forecasting model.
----------------------------------------------------------------------------

WHY THIS EXISTS:
  The old evaluation was a single 70/15/15 split, scored with MAPE (a
  metric that divides by actual output — which goes to zero at dawn/dusk,
  making the error explode or vanish depending on luck), and reported no
  baseline at all. A model that "beats" nothing but its own training
  noise isn't a forecasting result (audit finding F1).

  This script does two separate things:

  1. ROLLING-ORIGIN BACKTEST — walk forward through the development
     period, retraining fresh at each origin on only the data before it,
     and scoring the next 24 hours. This replaces the single static split
     with something that actually resembles how the model would be used:
     trained on the past, tested on the future, repeatedly.

  2. FINAL HOLDOUT — the one number that matters. Scored exactly once,
     on a period generated with a different date range AND a different
     random seed than anything src/models/train.py or any test has ever
     touched. This is the number you're allowed to trust.

  Every score is nMAE / nRMSE as a PERCENTAGE OF THE PLANT'S AC CAPACITY
  (never MAPE), against three baselines with zero learned parameters:

    - persistence          forecast(t) = actual(t - 24h)
    - smart_persistence    forecast(t) = clear_sky_power(t) *
                                          [actual(t-24h) / clear_sky_power(t-24h)]
                            (persist YESTERDAY's clear-sky ratio)
    - physics               forecast(t) = forecast_GHI(t) -> power via the
                            plant's rated efficiency only, no learning at all

  Per roadmap rule: if the model doesn't beat smart_persistence on the
  final holdout, that is a real finding to stop and discuss — not
  something this script papers over.

RUN WITH:
  python benchmark.py
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.pipeline import build_features, SERVING_FEATURE_COLUMNS
from src.ingestion.jaipur_simulator import generate_jaipur_weather
from src.ingestion.synthetic_forecast import simulate_day_ahead_forecast
from src.utils.config_loader import get_config

MODEL_PATH = PROJECT_ROOT / "src" / "models" / "xgboost_solar_v2.pkl"

# Must match src/models/train.py exactly -- this IS the development period.
DEV_START, DEV_DAYS, DEV_SEED = "2024-01-01", 365, 42
DEV_FORECAST_NOISE_SEED = 7  # matches src/models/train.py's TRAINING_FORECAST_NOISE_SEED

# A period + seed neither train.py nor any test has ever generated. Must
# fall entirely outside [DEV_START, DEV_START + DEV_DAYS) -- a different
# YEAR (the simulator's seasonality depends only on calendar month, so a
# different year still exercises the same season, but with a fresh random
# weather draw the model has never seen).
HOLDOUT_START, HOLDOUT_DAYS, HOLDOUT_SEED = "2025-06-01", 30, 99

MIN_TRAIN_HOURS = 24 * 30   # at least 30 days of history before the first origin
ORIGIN_STRIDE_HOURS = 24 * 7  # weekly origins
FORECAST_HORIZON_HOURS = 24
LAG_HOURS = 24               # how far back persistence/smart-persistence look

BASELINE_NAMES = ["model", "persistence", "smart_persistence", "physics"]


def _physics_baseline(features: pd.DataFrame, config: dict) -> pd.Series:
    """No ML: forecast GHI -> power via the plant's rated conversion only."""
    plant = config["solar_plant"]
    power = (features["shortwave_radiation"] / 1000) * plant["performance_ratio"] * plant["capacity_mw"]
    return power.clip(lower=0, upper=plant["capacity_mw"])


def _clear_sky_power(features: pd.DataFrame, config: dict) -> pd.Series:
    plant = config["solar_plant"]
    return (features["clear_sky_ghi_model"] / 1000) * plant["performance_ratio"] * plant["capacity_mw"]


def _persistence_baseline(y: pd.Series) -> pd.Series:
    """D-1 same block: forecast(t) = actual(t - 24h)."""
    return y.shift(LAG_HOURS)


def _smart_persistence_baseline(y: pd.Series, features: pd.DataFrame, config: dict) -> pd.Series:
    """Persist YESTERDAY's clear-sky ratio, applied to TODAY's clear-sky
    estimate. Uses only y(t-24h) and clear_sky_power(t-24h)/(t) -- nothing
    at or after t, so this is a legitimate forecast, not a leak."""
    csp = _clear_sky_power(features, config)
    ratio_yesterday = (y / csp.replace(0, np.nan)).shift(LAG_HOURS)
    return (csp * ratio_yesterday).clip(lower=0)


def _nmae_nrmse(y_true: pd.Series, y_pred: pd.Series, capacity_mw: float,
                 daytime_mask: pd.Series):
    """Restricted to daylight hours, for every method equally. Nighttime
    power is trivially ~0 for any method, so including it would flatter
    everyone's score by the same amount EXCEPT smart_persistence, whose
    clear-sky ratio is undefined (0/0) at night and drops out on its own --
    scoring it against a daytime-only mask that every other method also
    uses keeps the comparison apples-to-apples."""
    mask = y_true.notna() & y_pred.notna() & daytime_mask
    err = y_true[mask] - y_pred[mask]
    mae = err.abs().mean()
    rmse = np.sqrt((err ** 2).mean())
    return 100 * mae / capacity_mw, 100 * rmse / capacity_mw, int(mask.sum())


def _fresh_model() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
    )


def rolling_origin_backtest(features: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Walk forward through the development period: at each origin, train
    only on the past, score the next FORECAST_HORIZON_HOURS. Baselines are
    parameter-free, so they're just evaluated on the same test window."""
    y = features["solar_output_mw"]
    capacity_mw = config["solar_plant"]["capacity_mw"]
    daytime = features["clear_sky_ghi_model"] > 1.0

    persistence = _persistence_baseline(y)
    smart_persistence = _smart_persistence_baseline(y, features, config)
    physics = _physics_baseline(features, config)

    origins = list(range(MIN_TRAIN_HOURS, len(features) - FORECAST_HORIZON_HOURS, ORIGIN_STRIDE_HOURS))
    assert len(origins) >= 8, f"only {len(origins)} origins -- widen the development period"

    rows = []
    for origin in origins:
        train = slice(0, origin)
        test = slice(origin, origin + FORECAST_HORIZON_HOURS)

        model = _fresh_model()
        model.fit(features[SERVING_FEATURE_COLUMNS].iloc[train], y.iloc[train])
        model_preds = pd.Series(
            np.clip(model.predict(features[SERVING_FEATURE_COLUMNS].iloc[test]), 0, capacity_mw),
            index=y.iloc[test].index,
        )

        row = {"origin": features.index[origin]}
        for name, preds in [
            ("model", model_preds), ("persistence", persistence.iloc[test]),
            ("smart_persistence", smart_persistence.iloc[test]), ("physics", physics.iloc[test]),
        ]:
            nmae, nrmse, n = _nmae_nrmse(y.iloc[test], preds, capacity_mw, daytime.iloc[test])
            row[f"{name}_nmae"], row[f"{name}_nrmse"] = nmae, nrmse
        rows.append(row)

    results = pd.DataFrame(rows)
    print(f"ROLLING-ORIGIN BACKTEST -- {len(origins)} origins, {DEV_START} + {DEV_DAYS}d development period")
    print(f"(retrained fresh at each origin on only the data before it, {FORECAST_HORIZON_HOURS}h horizon)\n")
    numeric_cols = [c for c in results.columns if c != "origin"]
    print(results.assign(**{c: results[c].round(2) for c in numeric_cols}).to_string(index=False))
    print("\nAveraged across all origins:")
    for name in BASELINE_NAMES:
        print(f"  {name:18s}  nMAE {results[f'{name}_nmae'].mean():6.2f}%   nRMSE {results[f'{name}_nrmse'].mean():6.2f}%")
    return results


def final_holdout_eval(served_model, config: dict) -> dict:
    """The one number that counts: the model actually being served, scored
    ONCE on a period it has never seen -- different date range, different
    random seed than train.py or any test file uses.

    Scored TWICE, per roadmap P0.5, because "the model" and "the product"
    have different error sources:
      - MODEL SKILL: fed the true weather the simulator generated, as if
        the weather forecast were perfect. Isolates how good the model
        itself is.
      - DELIVERABLE SKILL: fed a simulated D-1 09:00 forecast with realistic
        forecast error baked in (see _synthetic_day_ahead_forecast). This is
        what a real customer actually receives -- the gap between the two
        numbers IS the weather-forecast error, and reporting only the first
        one would be misleading.
    """
    raw = generate_jaipur_weather(start_date=HOLDOUT_START, days=HOLDOUT_DAYS, seed=HOLDOUT_SEED)
    capacity_mw = config["solar_plant"]["capacity_mw"]

    observed_features = build_features(raw, config)
    forecast_features = build_features(simulate_day_ahead_forecast(raw, seed=HOLDOUT_SEED + 1), config)
    # y and the daylight mask come from the true data in both cases -- only
    # the WEATHER INPUT fed to the model differs between the two scorings.
    y = observed_features["solar_output_mw"]
    daytime = observed_features["clear_sky_ghi_model"] > 1.0

    persistence = _persistence_baseline(y)
    smart_persistence = _smart_persistence_baseline(y, observed_features, config)
    physics = _physics_baseline(observed_features, config)

    scores = {}
    for label, features in [("MODEL SKILL (observed weather)", observed_features),
                             ("DELIVERABLE SKILL (D-1 09:00 forecast weather, synthetic)", forecast_features)]:
        model_preds = pd.Series(
            np.clip(served_model.predict(features[SERVING_FEATURE_COLUMNS]), 0, capacity_mw),
            index=features.index,
        )
        print(f"\nFINAL HOLDOUT -- {label}")
        print(f"({HOLDOUT_START} + {HOLDOUT_DAYS}d, seed={HOLDOUT_SEED} -- never used by "
              f"src/models/train.py or any test, scored once)\n")
        for name, preds in [
            ("model", model_preds), ("persistence", persistence),
            ("smart_persistence", smart_persistence), ("physics", physics),
        ]:
            nmae, nrmse, n = _nmae_nrmse(y, preds, capacity_mw, daytime)
            scores.setdefault(name, {})[label] = nmae
            print(f"  {name:18s}  nMAE {nmae:6.2f}%   nRMSE {nrmse:6.2f}%   (n={n})")

    model_skill = scores["model"]["MODEL SKILL (observed weather)"]
    deliverable_skill = scores["model"]["DELIVERABLE SKILL (D-1 09:00 forecast weather, synthetic)"]
    smart_skill = scores["smart_persistence"]["DELIVERABLE SKILL (D-1 09:00 forecast weather, synthetic)"]
    print(f"\nWeather-forecast error cost: {deliverable_skill - model_skill:+.2f} points of nMAE "
          f"(model skill {model_skill:.2f}% -> deliverable skill {deliverable_skill:.2f}%).")

    if deliverable_skill < smart_skill:
        print(f"PASS -- deliverable skill ({deliverable_skill:.2f}% nMAE) beats smart_persistence "
              f"({smart_skill:.2f}% nMAE) under realistic forecast-weather conditions.")
    else:
        print(f"FAIL -- deliverable skill ({deliverable_skill:.2f}% nMAE) does NOT beat "
              f"smart_persistence ({smart_skill:.2f}% nMAE) once forecast-weather error is "
              f"accounted for. Per roadmap P0.4: stop and diagnose before building anything else.")
    return scores


def main():
    config = get_config()

    if not MODEL_PATH.exists():
        raise SystemExit(f"No served model at {MODEL_PATH} -- run `python -m src.models.train` first.")
    with open(MODEL_PATH, "rb") as f:
        served_model = pickle.load(f)

    dev_raw = generate_jaipur_weather(start_date=DEV_START, days=DEV_DAYS, seed=DEV_SEED)
    # Forecast-quality inputs, matching src/models/train.py -- the served
    # model is trained this way now (P0.5), so the backtest's own internal
    # retrains at each origin should reflect the same real-world condition,
    # not the simulator's perfect weather.
    dev_forecast_quality = simulate_day_ahead_forecast(dev_raw, seed=DEV_FORECAST_NOISE_SEED)
    dev_features = build_features(dev_forecast_quality, config)

    rolling_origin_backtest(dev_features, config)
    final_holdout_eval(served_model, config)


if __name__ == "__main__":
    main()
