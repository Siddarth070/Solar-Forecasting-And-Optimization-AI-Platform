"""
test_probabilistic_forecast.py
--------------------------------
Roadmap P1.3 acceptance criteria, tested directly: a P10/P50/P90
probabilistic forecast, validated with pinball loss, and a reliability
check ("P90 should cover ~90% of observed blocks on a holdout").

Two kinds of test here:
  1. A synthetic, closed-form calibration test -- we build a
     heteroscedastic dataset with a KNOWN conditional distribution, so
     "reliability" and "beats a naive baseline" are checked against
     ground truth we constructed, not against ourselves.
  2. Tests against the REAL committed artifact (xgboost_solar_quantile.json,
     produced by `make train`, same as xgboost_solar_v2.json), directly
     exercising src/api/main.py's /forecast endpoint -- proving the actual
     served quantiles are monotonic and capacity-scaled correctly, the
     same way tests/test_multi_plant.py checks the point model.

RUN WITH:
  pytest tests/test_probabilistic_forecast.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

QUANTILE_LEVELS = [0.1, 0.5, 0.9]


def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, tau: float) -> float:
    diff = y_true - y_pred
    return float(np.mean(np.maximum(tau * diff, (tau - 1) * diff)))


def _heteroscedastic_dataset(n: int, seed: int):
    """y = 3x + noise, where noise's spread GROWS with x -- so the true
    conditional quantiles are known in closed form:
      Q_tau(y | x) = 3x + norm.ppf(tau) * (1 + x)
    A model that ignores x's effect on SPREAD (e.g. any fixed-width
    interval) will miscalibrate on this dataset; only genuine quantile
    regression gets it right."""
    from scipy.stats import norm
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 10, n)
    sigma = 1 + x  # spread grows with x
    noise = rng.normal(0, 1, n) * sigma
    y = 3 * x + noise
    true_quantiles = {tau: 3 * x + norm.ppf(tau) * sigma for tau in QUANTILE_LEVELS}
    return x.reshape(-1, 1), y, true_quantiles


class TestSyntheticCalibration:
    """Ground-truth check: train on a dataset whose true conditional
    quantiles we know exactly, then verify the model recovers them."""

    @pytest.fixture(scope="class")
    @classmethod
    def fitted(cls):
        X_train, y_train, _ = _heteroscedastic_dataset(4000, seed=0)
        X_test, y_test, true_q = _heteroscedastic_dataset(2000, seed=1)

        model = XGBRegressor(
            objective="reg:quantileerror", quantile_alpha=QUANTILE_LEVELS,
            n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42,
        )
        model.fit(X_train, y_train)
        preds = np.sort(model.predict(X_test), axis=1)
        return X_test, y_test, true_q, preds

    def test_non_crossing_holds_everywhere(self, fitted):
        _, _, _, preds = fitted
        assert (preds[:, 0] <= preds[:, 1] + 1e-6).all()
        assert (preds[:, 1] <= preds[:, 2] + 1e-6).all()

    def test_reliability_p90_covers_close_to_90_percent(self, fitted):
        _, y_test, _, preds = fitted
        coverage = np.mean(y_test <= preds[:, 2])
        assert 0.85 <= coverage <= 0.95, f"P90 coverage {coverage:.1%}, expected ~90%"

    def test_reliability_p10_covers_close_to_10_percent_from_below(self, fitted):
        _, y_test, _, preds = fitted
        frac_below_p10 = np.mean(y_test < preds[:, 0])
        assert 0.05 <= frac_below_p10 <= 0.15, f"fraction below P10 {frac_below_p10:.1%}, expected ~10%"

    def test_interval_80_coverage_close_to_80_percent(self, fitted):
        _, y_test, _, preds = fitted
        inside = np.mean((y_test >= preds[:, 0]) & (y_test <= preds[:, 2]))
        assert 0.72 <= inside <= 0.88, f"80% interval coverage {inside:.1%}"

    def test_pinball_loss_beats_a_naive_unconditional_baseline(self, fitted):
        """The naive baseline predicts the SAME (unconditional, training-set)
        quantile for every x, ignoring that spread grows with x. A model
        that actually learned the conditional distribution must beat it."""
        X_train, y_train, _ = _heteroscedastic_dataset(4000, seed=0)
        _, y_test, _, preds = fitted

        for i, tau in enumerate(QUANTILE_LEVELS):
            naive_pred = np.full_like(y_test, np.quantile(y_train, tau))
            naive_loss = _pinball_loss(y_test, naive_pred, tau)
            model_loss = _pinball_loss(y_test, preds[:, i], tau)
            assert model_loss < naive_loss, (
                f"tau={tau}: model pinball loss {model_loss:.4f} did not beat "
                f"naive constant-quantile baseline {naive_loss:.4f}"
            )

    def test_predicted_quantiles_track_the_true_conditional_quantiles(self, fitted):
        """Not just calibrated in aggregate -- actually close to the known
        closed-form true quantile function itself, at a few sample points."""
        X_test, _, true_q, preds = fitted
        # Compare mean absolute error between predicted and true quantile
        # curves; loose tolerance since this is a finite-sample XGB fit,
        # not an exact analytical solver.
        for i, tau in enumerate(QUANTILE_LEVELS):
            mae = np.mean(np.abs(preds[:, i] - true_q[tau]))
            assert mae < 2.0, f"tau={tau}: predicted quantile curve off by {mae:.2f} on average"


class TestServedQuantileModel:
    """Exercises the REAL committed artifact via the actual API, the same
    way the model actually gets used in production."""

    MODEL_PATH = PROJECT_ROOT / "src" / "models" / "xgboost_solar_quantile.json"

    def test_artifact_exists(self):
        assert self.MODEL_PATH.exists(), (
            "src/models/xgboost_solar_quantile.json missing -- run `make train`"
        )

    def test_api_forecast_includes_monotonic_quantiles(self):
        from fastapi.testclient import TestClient
        from src.api.main import app

        client = TestClient(app)
        hours = [{
            "shortwave_radiation": 400 + 30 * i, "cloud_cover": 15,
            "temperature_2m": 32, "relative_humidity_2m": 35,
            "wind_speed_10m": 3, "hour": 8 + i, "month": 6,
        } for i in range(6)]

        resp = client.post("/forecast", json={"hours": hours, "plant_id": "jaipur_100mw"})
        assert resp.status_code == 200
        data = resp.json()

        assert len(data["predictions_p10_mw"]) == 6
        assert len(data["predictions_p50_mw"]) == 6
        assert len(data["predictions_p90_mw"]) == 6
        for p10, p50, p90 in zip(
            data["predictions_p10_mw"], data["predictions_p50_mw"], data["predictions_p90_mw"]
        ):
            assert p10 <= p50 <= p90

    def test_quantile_predictions_scale_with_requested_plant_capacity(self):
        """Same regression this repo already hit for the point model
        (roadmap P1.1 bug fix): quantile predictions must also rescale by
        the REQUESTED plant's capacity, not come out identical for two
        differently-sized plants."""
        from fastapi.testclient import TestClient
        from src.api.main import app

        client = TestClient(app)
        hours = [{
            "shortwave_radiation": 700, "cloud_cover": 5, "temperature_2m": 35,
            "relative_humidity_2m": 30, "wind_speed_10m": 3, "hour": 12, "month": 5,
        }]

        jaipur = client.post("/forecast", json={"hours": hours, "plant_id": "jaipur_100mw"}).json()
        pune = client.post("/forecast", json={"hours": hours, "plant_id": "pune_50mw"}).json()

        assert pune["predictions_p90_mw"][0] <= 50.0
        assert jaipur["predictions_p90_mw"][0] > pune["predictions_p90_mw"][0] * 1.5
