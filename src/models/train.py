"""
train.py — Train the served XGBoost solar-forecast model.
------------------------------------------------------------

WHY THIS EXISTS:
  xgboost_solar_v2.pkl used to be produced by no code in this repo — the
  only training notebook built a different, 25-feature version, and the
  notebook itself had hardcoded local paths (audit finding F3). This
  script is the first reproducible path from data to served artifact.

  It is a MINIMAL trainer, not the full reproducibility story: no
  `make train` target, no model_card.json, no JSON serialization, no
  git-SHA tracking, no proper rolling-origin evaluation with baselines.
  Those are roadmap task P0.6 (reproducibility) and P0.4 (honest
  evaluation harness) — deliberately not done here. This script exists
  so that P0.2 (single feature pipeline, no hand-duplicated feature
  dicts in the API/dashboard) has a real model to serve that actually
  matches src.features.pipeline's feature names and values, instead of
  the previous model, which was trained on a leaky clear_sky_ratio
  feature under different column names entirely.

WHAT IT TRAINS ON:
  Only SERVING_FEATURE_COLUMNS — weather + time-of-day + clear_sky_index.
  Lag/rolling features are deliberately excluded: a live dashboard/API
  request is a one-shot weather forecast with no access to the plant's
  actual past generation, so a model trained to expect solar_lag_1h etc.
  could never be honestly served (see src/features/pipeline.py).

RUN WITH:
  python -m src.models.train
"""

import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.pipeline import build_features, SERVING_FEATURE_COLUMNS
from src.ingestion.jaipur_simulator import generate_jaipur_weather
from src.utils.config_loader import get_config

MODEL_PATH = PROJECT_ROOT / "src" / "models" / "xgboost_solar_v2.pkl"
TARGET_COLUMN = "solar_output_mw"


def train():
    config = get_config()

    raw = generate_jaipur_weather(start_date="2024-01-01", days=90)
    featured = build_features(raw, config).dropna(
        subset=[*SERVING_FEATURE_COLUMNS, TARGET_COLUMN]
    )

    # Simple chronological split — a placeholder honesty check only.
    # The real evaluation (rolling-origin backtest, persistence and
    # smart-persistence baselines, nMAE/nRMSE as % of AC capacity) is
    # roadmap task P0.4, deliberately not built here.
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

    predictions = np.clip(model.predict(X_test), 0, None)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    capacity_mw = config["solar_plant"]["capacity_mw"]

    print(f"Trained on {len(X_train)} rows, held out {len(X_test)} rows (last 20%, chronological).")
    print(f"[PROVISIONAL — see P0.4 for the real evaluation harness]")
    print(f"  MAE:  {mae:.2f} MW  ({100 * mae / capacity_mw:.2f}% of {capacity_mw:.0f} MW capacity)")
    print(f"  RMSE: {rmse:.2f} MW ({100 * rmse / capacity_mw:.2f}% of {capacity_mw:.0f} MW capacity)")

    importances = sorted(
        zip(SERVING_FEATURE_COLUMNS, model.feature_importances_),
        key=lambda kv: kv[1], reverse=True,
    )
    print("Feature importances:")
    for name, score in importances:
        print(f"  {name:25s} {score:.4f}")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    train()
