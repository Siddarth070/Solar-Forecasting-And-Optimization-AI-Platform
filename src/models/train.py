"""
train.py — Train the served XGBoost solar-forecast model.
------------------------------------------------------------

WHY THIS EXISTS:
  xgboost_solar_v2.pkl used to be produced by no code in this repo — the
  only training notebook built a different, 25-feature version, and the
  notebook itself had hardcoded local paths (audit finding F3). This
  script is the reproducible path from data to served artifact: `make
  train` (see the repo-root Makefile) runs it end to end on a clean
  clone, with no manual steps.

WHAT IT TRAINS ON:
  Only SERVING_FEATURE_COLUMNS — weather + time-of-day + clear_sky_index.
  Lag/rolling features are deliberately excluded: a live dashboard/API
  request is a one-shot weather forecast with no access to the plant's
  actual past generation, so a model trained to expect solar_lag_1h etc.
  could never be honestly served (see src/features/pipeline.py).

  The WEATHER INPUTS are forecast-quality, not perfect (see
  src/ingestion/synthetic_forecast.py) — a live request only ever supplies
  a forecast, never the true value, so training on the simulator's perfect
  weather would teach the model a distribution it will never actually see
  in production. (benchmark.py's P0.5 "deliverable skill" scoring caught
  exactly this mismatch: a model trained on true weather looked excellent
  fed true weather and collapsed below a trivial baseline fed realistic
  forecast weather.) The TARGET is untouched — solar_output_mw is what
  really generated; only the weather the model gets to see is degraded.

  The training data itself is not stored anywhere (data/ is gitignored
  and empty) — it doesn't need to be: generate_jaipur_weather() plus
  simulate_day_ahead_forecast() are deterministic given the seeds
  recorded below and in model_card.json, so anyone can rebuild the exact
  training set from this script alone.

KNOWN LIMITATION (roadmap P1.1): this script trains against ONE plant's
  config (TRAINING_PLANT_ID), and location/capacity now come from
  configs/plants/<id>.yaml rather than a hardcoded global. But the
  underlying weather SIMULATOR (generate_jaipur_weather) is still
  Jaipur-specific — its seasonal/diurnal formulas are calibrated to
  Jaipur's actual climate, not generic. Pointing TRAINING_PLANT_ID at a
  different plant (e.g. pune_50mw) would train on that plant's real
  location/capacity but Jaipur's synthetic weather patterns, which is
  not a real Pune model. A location-aware simulator, or real per-plant
  data (roadmap Phase 3), is needed before that's honest.

REPRODUCIBILITY:
  The model is saved as JSON (XGBoost's native format, human-diffable),
  not pickle — a pickle can silently break across library versions and
  is opaque to review. src/models/model_card.json is written alongside
  it with the git commit, data window, feature list, and this script's
  own quick sanity-check metrics. benchmark.py (P0.4/P0.5) later adds
  the real evaluation — rolling-origin backtest, baselines, and the
  model-skill/deliverable-skill split — to the SAME file.

ALSO TRAINS (roadmap P1.3): a second, PROBABILISTIC model producing P10/P50/
  P90 quantile forecasts (xgboost_solar_quantile.json) -- same features,
  same capacity-fraction target, same train/test rows as the point model
  above, differing only in objective (multi-output `reg:quantileerror`).
  Sharing the exact same data means the two models are directly comparable
  (the quantile model's P50 should track the point model closely) and
  there is no second, silently-diverging data pipeline to keep in sync.
  See benchmark.py's pinball-loss + reliability evaluation on the final
  holdout for the real accuracy check; the metrics printed here are a
  provisional in-sample sanity check only, same as the point model's.

RUN WITH:
  make train
  (or directly: python -m src.models.train)
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.pipeline import build_features, SERVING_FEATURE_COLUMNS
from src.ingestion.jaipur_simulator import generate_jaipur_weather
from src.ingestion.synthetic_forecast import simulate_day_ahead_forecast
from src.models.model_card import update_model_card
from src.utils.config_loader import get_plant_config

MODEL_PATH = PROJECT_ROOT / "src" / "models" / "xgboost_solar_v2.json"
QUANTILE_MODEL_PATH = PROJECT_ROOT / "src" / "models" / "xgboost_solar_quantile.json"
QUANTILE_LEVELS = [0.1, 0.5, 0.9]
TARGET_COLUMN = "solar_output_mw"

TRAINING_PLANT_ID = "jaipur_100mw"  # see KNOWN LIMITATION above
TRAINING_START = "2024-01-01"
TRAINING_DAYS = 365
TRAINING_WEATHER_SEED = 42  # generate_jaipur_weather's own default
TRAINING_FORECAST_NOISE_SEED = 7  # distinct from benchmark.py's holdout seeds (99, 100)


def train():
    plant_config = get_plant_config(TRAINING_PLANT_ID)
    capacity_mw = plant_config["capacity"]["ac_capacity_mw"]

    true_weather = generate_jaipur_weather(
        start_date=TRAINING_START, days=TRAINING_DAYS, seed=TRAINING_WEATHER_SEED
    )
    forecast_quality = simulate_day_ahead_forecast(true_weather, seed=TRAINING_FORECAST_NOISE_SEED)
    featured = build_features(forecast_quality, plant_config).dropna(
        subset=[*SERVING_FEATURE_COLUMNS, TARGET_COLUMN]
    )
    # Train on CAPACITY FACTOR (0-1, target / this plant's AC capacity), not
    # raw MW. A model trained on absolute MW learns Jaipur's specific scale
    # and produces nonsense when served for a differently-sized plant (found
    # by actually testing pune_50mw through the API: it predicted ~49 MW,
    # nearly identical to jaipur_100mw's ~49 MW, because the model has no
    # notion of "this plant's capacity" — only clipping at the ceiling
    # papers over it). Serving code multiplies back by the REQUESTED plant's
    # capacity (src/api/main.py, dashboard/app.py), so one model genuinely
    # generalizes across differently-sized plants.
    featured[TARGET_COLUMN] = featured[TARGET_COLUMN] / capacity_mw

    # Simple chronological split — a placeholder honesty check only. The
    # real evaluation (rolling-origin backtest, baselines, nMAE/nRMSE,
    # model-skill/deliverable-skill split) lives in benchmark.py.
    split = int(len(featured) * 0.8)
    train_df, test_df = featured.iloc[:split], featured.iloc[split:]

    X_train, y_train = train_df[SERVING_FEATURE_COLUMNS], train_df[TARGET_COLUMN]
    X_test, y_test = test_df[SERVING_FEATURE_COLUMNS], test_df[TARGET_COLUMN]

    model = XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # y_test/predictions are capacity fraction (0-1); nMAE/nRMSE as a % of
    # capacity is just that fraction * 100 — no separate division needed.
    predictions = np.clip(model.predict(X_test), 0, None)
    nmae_pct = 100 * mean_absolute_error(y_test, predictions)
    nrmse_pct = 100 * np.sqrt(mean_squared_error(y_test, predictions))

    print(f"Trained on {len(X_train)} rows, held out {len(X_test)} rows (last 20%, chronological).")
    print("[PROVISIONAL split -- see benchmark.py / model_card.json for the real evaluation]")
    print(f"  nMAE:  {nmae_pct:.2f}% of {capacity_mw:.0f} MW capacity")
    print(f"  nRMSE: {nrmse_pct:.2f}% of {capacity_mw:.0f} MW capacity")

    importances = sorted(
        zip(SERVING_FEATURE_COLUMNS, model.feature_importances_),
        key=lambda kv: kv[1], reverse=True,
    )
    print("Feature importances:")
    for name, score in importances:
        print(f"  {name:25s} {score:.4f}")

    model.save_model(str(MODEL_PATH))
    print(f"Saved model to {MODEL_PATH}")

    # ── Probabilistic model (roadmap P1.3): same rows, same target, same
    # features -- only the objective changes. XGBoost's multi-output
    # `reg:quantileerror` trains all three quantiles in one booster, which
    # also keeps P10 <= P50 <= P90 non-crossing by construction far more
    # reliably than three independently-fit models would.
    quantile_model = XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=QUANTILE_LEVELS,
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    quantile_model.fit(X_train, y_train)

    q_preds = np.clip(quantile_model.predict(X_test), 0, None)
    q_preds = np.sort(q_preds, axis=1)  # defensive: enforce non-crossing
    pinball_by_tau = {}
    for i, tau in enumerate(QUANTILE_LEVELS):
        diff = y_test.to_numpy() - q_preds[:, i]
        pinball_by_tau[str(tau)] = round(
            100 * float(np.mean(np.maximum(tau * diff, (tau - 1) * diff))), 4
        )
    print(f"\n[Quantile model, PROVISIONAL split] Pinball loss (x100, capacity-fraction units) "
          f"by quantile: {pinball_by_tau}")

    quantile_model.save_model(str(QUANTILE_MODEL_PATH))
    print(f"Saved quantile model to {QUANTILE_MODEL_PATH}")

    card = update_model_card(
        model_path=MODEL_PATH.name,
        plant_id=TRAINING_PLANT_ID,
        data_window={
            "start": TRAINING_START,
            "days": TRAINING_DAYS,
            "weather_seed": TRAINING_WEATHER_SEED,
            "forecast_noise_seed": TRAINING_FORECAST_NOISE_SEED,
            "source": "src.ingestion.jaipur_simulator.generate_jaipur_weather "
                      "+ src.ingestion.synthetic_forecast.simulate_day_ahead_forecast "
                      "(synthetic, deterministic given these seeds -- not stored, regenerable "
                      "by re-running `make train`)",
        },
        feature_list=SERVING_FEATURE_COLUMNS,
        target="capacity_factor (solar_output_mw / plant's ac_capacity_mw, range 0-1) "
               "-- NOT raw MW. Serving code must multiply predictions by the REQUESTED "
               "plant's ac_capacity_mw. This is what lets one model serve multiple "
               "differently-sized plants (roadmap P1.1).",
        train_test_split_metrics={
            "split": "80/20 chronological, provisional only",
            "n_train": len(X_train),
            "n_test": len(X_test),
            "nmae_pct": round(nmae_pct, 3),
            "nrmse_pct": round(nrmse_pct, 3),
            "note": "not the real evaluation -- see benchmark.py's rolling-origin backtest "
                    "and final holdout in this same file, under rolling_origin_backtest / "
                    "final_holdout",
        },
        quantile_model={
            "model_path": QUANTILE_MODEL_PATH.name,
            "quantile_levels": QUANTILE_LEVELS,
            "objective": "reg:quantileerror (single multi-output booster, non-crossing by construction)",
            "target": "capacity_factor, same as the point model above",
            "provisional_split_pinball_loss_x100": {
                "split": "80/20 chronological, provisional only, same rows as the point model",
                "note": "not the real evaluation -- see benchmark.py's quantile_holdout_eval "
                        "in this same file, under quantile_final_holdout, for pinball loss "
                        "and reliability (P10/P50/P90 coverage) on a real holdout",
                **pinball_by_tau,
            },
        },
    )
    print(f"Updated model card for {plant_config['name']} at src/models/model_card.json "
          f"(git_sha={card['git_sha'][:12]})")


if __name__ == "__main__":
    train()
