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

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.baselines import (
    clear_sky_power as _clear_sky_power,
    nmae_nrmse as _nmae_nrmse,
    persistence_baseline as _persistence_baseline,
    physics_baseline as _physics_baseline,
    smart_persistence_baseline as _smart_persistence_baseline,
)
from src.features.pipeline import build_features, SERVING_FEATURE_COLUMNS
from src.ingestion.jaipur_simulator import generate_jaipur_weather
from src.ingestion.synthetic_forecast import simulate_day_ahead_forecast
from src.models.model_card import update_model_card
from src.utils.config_loader import get_plant_config

MODEL_PATH = PROJECT_ROOT / "src" / "models" / "xgboost_solar_v2.json"
QUANTILE_MODEL_PATH = PROJECT_ROOT / "src" / "models" / "xgboost_solar_quantile.json"
QUANTILE_LEVELS = [0.1, 0.5, 0.9]
BENCHMARK_PLANT_ID = "jaipur_100mw"  # must match src/models/train.py's TRAINING_PLANT_ID --
                                     # scoring the served model against a plant it wasn't
                                     # trained for would be meaningless

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
# LAG_HOURS lives in src.evaluation.baselines now (how far back
# persistence/smart-persistence look) -- kept in one place, not
# duplicated here, so it can't drift from what those functions actually use.

BASELINE_NAMES = ["model", "persistence", "smart_persistence", "physics"]

# _physics_baseline, _clear_sky_power, _persistence_baseline,
# _smart_persistence_baseline and _nmae_nrmse now live in
# src/evaluation/baselines.py (imported above), so roadmap P2.9's weekly
# production report scores against the exact same baseline math, not a
# separate implementation that could quietly drift from this one.


def _fresh_model() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
    )


def rolling_origin_backtest(features: pd.DataFrame, plant_config: dict) -> pd.DataFrame:
    """Walk forward through the development period: at each origin, train
    only on the past, score the next FORECAST_HORIZON_HOURS. Baselines are
    parameter-free, so they're just evaluated on the same test window."""
    y = features["solar_output_mw"]
    capacity_mw = plant_config["capacity"]["ac_capacity_mw"]
    daytime = features["clear_sky_ghi_model"] > 1.0

    persistence = _persistence_baseline(y)
    smart_persistence = _smart_persistence_baseline(y, features, plant_config)
    physics = _physics_baseline(features, plant_config)

    origins = list(range(MIN_TRAIN_HOURS, len(features) - FORECAST_HORIZON_HOURS, ORIGIN_STRIDE_HOURS))
    assert len(origins) >= 8, f"only {len(origins)} origins -- widen the development period"

    rows = []
    for origin in origins:
        train = slice(0, origin)
        test = slice(origin, origin + FORECAST_HORIZON_HOURS)

        # Train on capacity FRACTION, matching src/models/train.py, so this
        # internal diagnostic model reflects the same methodology as the
        # actually-served model, not a different convention.
        model = _fresh_model()
        model.fit(features[SERVING_FEATURE_COLUMNS].iloc[train], y.iloc[train] / capacity_mw)
        model_preds = pd.Series(
            np.clip(model.predict(features[SERVING_FEATURE_COLUMNS].iloc[test]) * capacity_mw, 0, capacity_mw),
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


def final_holdout_eval(served_model, plant_config: dict) -> dict:
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
    capacity_mw = plant_config["capacity"]["ac_capacity_mw"]

    observed_features = build_features(raw, plant_config)
    forecast_features = build_features(simulate_day_ahead_forecast(raw, seed=HOLDOUT_SEED + 1), plant_config)
    # y and the daylight mask come from the true data in both cases -- only
    # the WEATHER INPUT fed to the model differs between the two scorings.
    y = observed_features["solar_output_mw"]
    daytime = observed_features["clear_sky_ghi_model"] > 1.0

    persistence = _persistence_baseline(y)
    smart_persistence = _smart_persistence_baseline(y, observed_features, plant_config)
    physics = _physics_baseline(observed_features, plant_config)

    scores = {}
    for label, features in [("MODEL SKILL (observed weather)", observed_features),
                             ("DELIVERABLE SKILL (D-1 09:00 forecast weather, synthetic)", forecast_features)]:
        # served_model outputs capacity FRACTION (0-1), not MW -- rescale
        # by this plant's capacity (see src/models/train.py).
        model_preds = pd.Series(
            np.clip(served_model.predict(features[SERVING_FEATURE_COLUMNS]) * capacity_mw, 0, capacity_mw),
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


def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, tau: float) -> float:
    diff = y_true - y_pred
    return float(np.mean(np.maximum(tau * diff, (tau - 1) * diff)))


def quantile_final_holdout_eval(served_quantile_model, plant_config: dict) -> dict:
    """Roadmap P1.3 acceptance criterion: pinball loss and a reliability
    check (does P90 actually cover ~90% of observed blocks?), on the SAME
    final-holdout period and SAME model-skill/deliverable-skill split as
    final_holdout_eval above -- scored once, never touched by training.

    Reliability is checked two ways, both against the daytime-only mask
    used everywhere else in this file (nighttime output is trivially ~0
    for every quantile, which would flatter coverage without meaning
    anything):
      - coverage_p90: fraction of true values <= the P90 prediction.
        Should be close to 90% -- much lower means P90 is too tight
        (understates risk of a high-output block); much higher means it's
        too loose (uselessly wide).
      - coverage_p10: fraction of true values >= the P10 prediction.
        Should be close to 10%... i.e. fraction >= P10 close to 90%? No --
        by definition P10 should be exceeded by ~90% of true values, so
        we report fraction(y >= p10) and expect it near 90%, symmetric
        with P90's near-90% framing.
      - interval_80_coverage: fraction inside [P10, P90], expected ~80%.
    """
    raw = generate_jaipur_weather(start_date=HOLDOUT_START, days=HOLDOUT_DAYS, seed=HOLDOUT_SEED)
    capacity_mw = plant_config["capacity"]["ac_capacity_mw"]

    observed_features = build_features(raw, plant_config)
    forecast_features = build_features(simulate_day_ahead_forecast(raw, seed=HOLDOUT_SEED + 1), plant_config)
    y = observed_features["solar_output_mw"]
    daytime = (observed_features["clear_sky_ghi_model"] > 1.0).to_numpy()

    results = {}
    for label, features in [("MODEL SKILL (observed weather)", observed_features),
                             ("DELIVERABLE SKILL (D-1 09:00 forecast weather, synthetic)", forecast_features)]:
        # Quantile model outputs capacity FRACTION, three columns [P10,P50,P90]
        # (see src/models/train.py) -- rescale, then sort defensively so a
        # rare crossing never produces a nonsensical P10 > P90 interval.
        raw_preds = np.clip(served_quantile_model.predict(features[SERVING_FEATURE_COLUMNS]) * capacity_mw,
                             0, capacity_mw)
        preds = np.sort(raw_preds, axis=1)
        p10, p50, p90 = preds[:, 0], preds[:, 1], preds[:, 2]

        y_arr = y.to_numpy()
        mask = daytime & ~np.isnan(y_arr)
        y_m, p10_m, p50_m, p90_m = y_arr[mask], p10[mask], p50[mask], p90[mask]

        pinball = {
            str(tau): 100 * _pinball_loss(y_m, pred, tau) / capacity_mw
            for tau, pred in zip(QUANTILE_LEVELS, [p10_m, p50_m, p90_m])
        }
        coverage_p90 = float(np.mean(y_m <= p90_m))
        coverage_p10 = float(np.mean(y_m >= p10_m))
        interval_80_coverage = float(np.mean((y_m >= p10_m) & (y_m <= p90_m)))
        non_crossing = bool(np.all(preds[:, 0] <= preds[:, 1] + 1e-6) and np.all(preds[:, 1] <= preds[:, 2] + 1e-6))

        print(f"\nQUANTILE FINAL HOLDOUT -- {label}")
        print(f"({HOLDOUT_START} + {HOLDOUT_DAYS}d, seed={HOLDOUT_SEED}, n={mask.sum()} daytime blocks)\n")
        print(f"  Pinball loss (% of capacity): P10={pinball['0.1']:.3f}  P50={pinball['0.5']:.3f}  P90={pinball['0.9']:.3f}")
        print(f"  Reliability: P(y<=P90)={coverage_p90:.1%} (target ~90%)   "
              f"P(y>=P10)={coverage_p10:.1%} (target ~90%)   "
              f"P(P10<=y<=P90)={interval_80_coverage:.1%} (target ~80%)")
        print(f"  Non-crossing (P10<=P50<=P90) holds for all rows: {non_crossing}")

        results[label] = {
            "pinball_loss_pct_capacity": {k: round(v, 4) for k, v in pinball.items()},
            "coverage_p90": round(coverage_p90, 4),
            "coverage_p10": round(coverage_p10, 4),
            "interval_80_coverage": round(interval_80_coverage, 4),
            "non_crossing_holds": non_crossing,
        }

    deliverable = results["DELIVERABLE SKILL (D-1 09:00 forecast weather, synthetic)"]
    cov90, cov10, cov80 = (deliverable["coverage_p90"], deliverable["coverage_p10"],
                            deliverable["interval_80_coverage"])
    # A loose, honest tolerance band (+-10 points) -- this is a calibration
    # check on a single 30-day holdout, not a guarantee to the decimal.
    if 0.80 <= cov90 <= 0.98 and 0.80 <= cov10 <= 0.98 and 0.65 <= cov80 <= 0.92:
        print(f"\nPASS -- P10/P50/P90 are reasonably well-calibrated on the deliverable-skill holdout "
              f"(P90 coverage {cov90:.1%}, P10 coverage {cov10:.1%}, 80% interval coverage {cov80:.1%}).")
    else:
        print(f"\nFAIL -- quantile calibration is off on the deliverable-skill holdout "
              f"(P90 coverage {cov90:.1%}, P10 coverage {cov10:.1%}, 80% interval coverage {cov80:.1%}) "
              f"-- outside the expected ~80-98% / ~80-98% / ~65-92% bands. Investigate before trusting "
              f"these intervals.")
    return results


def main():
    plant_config = get_plant_config(BENCHMARK_PLANT_ID)

    if not MODEL_PATH.exists():
        raise SystemExit(f"No served model at {MODEL_PATH} -- run `make train` first.")
    served_model = XGBRegressor()
    served_model.load_model(str(MODEL_PATH))

    dev_raw = generate_jaipur_weather(start_date=DEV_START, days=DEV_DAYS, seed=DEV_SEED)
    # Forecast-quality inputs, matching src/models/train.py -- the served
    # model is trained this way now (P0.5), so the backtest's own internal
    # retrains at each origin should reflect the same real-world condition,
    # not the simulator's perfect weather.
    dev_forecast_quality = simulate_day_ahead_forecast(dev_raw, seed=DEV_FORECAST_NOISE_SEED)
    dev_features = build_features(dev_forecast_quality, plant_config)

    rolling_results = rolling_origin_backtest(dev_features, plant_config)
    holdout_scores = final_holdout_eval(served_model, plant_config)

    quantile_holdout_results = None
    if QUANTILE_MODEL_PATH.exists():
        served_quantile_model = XGBRegressor()
        served_quantile_model.load_model(str(QUANTILE_MODEL_PATH))
        quantile_holdout_results = quantile_final_holdout_eval(served_quantile_model, plant_config)
    else:
        print(f"\nNo quantile model at {QUANTILE_MODEL_PATH} -- skipping P1.3 pinball/reliability eval "
              f"(run `make train` first).")

    observed_label = "MODEL SKILL (observed weather)"
    forecast_label = "DELIVERABLE SKILL (D-1 09:00 forecast weather, synthetic)"
    update_model_card(
        rolling_origin_backtest={
            "n_origins": len(rolling_results),
            "horizon_hours": FORECAST_HORIZON_HOURS,
            "development_period": {"start": DEV_START, "days": DEV_DAYS},
            "averaged_nmae_pct": {
                name: round(float(rolling_results[f"{name}_nmae"].mean()), 3) for name in BASELINE_NAMES
            },
            "averaged_nrmse_pct": {
                name: round(float(rolling_results[f"{name}_nrmse"].mean()), 3) for name in BASELINE_NAMES
            },
        },
        final_holdout={
            "period": {"start": HOLDOUT_START, "days": HOLDOUT_DAYS, "seed": HOLDOUT_SEED},
            "model_skill_nmae_pct": {
                name: round(float(scores[observed_label]), 3) for name, scores in holdout_scores.items()
            },
            "deliverable_skill_nmae_pct": {
                name: round(float(scores[forecast_label]), 3) for name, scores in holdout_scores.items()
            },
        },
        **({"quantile_final_holdout": {
            "period": {"start": HOLDOUT_START, "days": HOLDOUT_DAYS, "seed": HOLDOUT_SEED},
            "quantile_levels": QUANTILE_LEVELS,
            "model_skill": quantile_holdout_results["MODEL SKILL (observed weather)"],
            "deliverable_skill": quantile_holdout_results[
                "DELIVERABLE SKILL (D-1 09:00 forecast weather, synthetic)"],
        }} if quantile_holdout_results is not None else {}),
    )
    print("\nUpdated src/models/model_card.json with the benchmark results above.")


if __name__ == "__main__":
    main()
