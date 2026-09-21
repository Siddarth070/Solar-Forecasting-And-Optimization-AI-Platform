"""
test_feature_parity.py
------------------------
Guards against train/serve skew (audit finding F2): the served model must
always expect exactly the columns build_features() actually produces for
a live request, in the same order. XGBoost's sklearn API already raises
if the names mismatch (see src/api/main.py / dashboard/app.py, which both
call build_features(...)[SERVING_FEATURE_COLUMNS] before predict()) — this
test exists so that a mismatch is caught by `pytest` in CI, on the very
next commit that touches either side, rather than as a 500 in production.

RUN WITH:
  pytest tests/test_feature_parity.py -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.pipeline import build_features, SERVING_FEATURE_COLUMNS
from src.utils.config_loader import get_config

MODEL_PATH = PROJECT_ROOT / "src" / "models" / "xgboost_solar_v2.json"


def _served_model():
    if not MODEL_PATH.exists():
        pytest.skip(f"served model not found at {MODEL_PATH}")
    model = XGBRegressor()
    model.load_model(str(MODEL_PATH))
    return model


def _representative_request_frame() -> pd.DataFrame:
    """A frame shaped exactly like what src/api/main.py and
    dashboard/app.py hand to build_features() for a live request: no
    solar_output_mw (a live weather forecast never has it)."""
    index = pd.date_range("2024-06-01", periods=24, freq="h", tz="Asia/Kolkata")
    return pd.DataFrame(
        {
            "hour": index.hour,
            "month": index.month,
            "shortwave_radiation": 500.0,
            "cloud_cover": 20.0,
            "temperature_2m": 32.0,
            "relative_humidity_2m": 35.0,
            "wind_speed_10m": 3.0,
        },
        index=index,
    )


class TestFeatureServingParity:
    def test_serving_columns_match_model_exactly_and_in_order(self):
        model = _served_model()
        config = get_config()

        features = build_features(_representative_request_frame(), config)
        served_columns = list(features[SERVING_FEATURE_COLUMNS].columns)

        assert served_columns == model.get_booster().feature_names, (
            "src.features.pipeline.SERVING_FEATURE_COLUMNS no longer matches "
            "what the served model expects (name or order changed on one "
            "side without the other). Retrain via `python -m src.models.train` "
            "after reconciling the two."
        )

    def test_no_serving_column_is_a_hardcoded_constant(self):
        """Every SERVING_FEATURE_COLUMNS value must actually vary with its
        input, not be a literal the serving code fabricated (audit finding
        F2 — 7 of 17 features used to be hardcoded to 0.0)."""
        config = get_config()

        # Span 12 different months at 12 different hours, so both the
        # hour-of-day and month-of-year cyclical encodings have room to
        # vary — a single day (as in the parity test above) would make
        # month_sin/cos falsely look "constant" here.
        index = pd.DatetimeIndex([
            pd.Timestamp(year=2024, month=m, day=15, hour=m, tz="Asia/Kolkata")
            for m in range(1, 13)
        ])
        n = len(index)
        varied_input = pd.DataFrame(
            {
                "hour": index.hour,
                "month": index.month,
                "shortwave_radiation": [100.0 + 20 * i for i in range(n)],
                "cloud_cover": [i * 3.0 for i in range(n)],
                "temperature_2m": [20.0 + 0.5 * i for i in range(n)],
                "relative_humidity_2m": [30.0 + i for i in range(n)],
                "wind_speed_10m": [1.0 + 0.2 * i for i in range(n)],
            },
            index=index,
        )

        features = build_features(varied_input, config)
        X = features[SERVING_FEATURE_COLUMNS]

        constant_columns = [col for col in X.columns if X[col].nunique() <= 1]
        assert not constant_columns, (
            f"these serving columns never change across a 24-hour request, "
            f"suggesting a hardcoded value rather than a real computation: "
            f"{constant_columns}"
        )

    def test_serving_columns_are_a_subset_of_the_full_feature_list(self):
        from src.features.pipeline import FEATURE_COLUMNS

        assert set(SERVING_FEATURE_COLUMNS).issubset(set(FEATURE_COLUMNS))
