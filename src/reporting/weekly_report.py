"""
weekly_report.py — Weekly forecast-performance report (roadmap P2.9).

WHY THIS EXISTS:
  A forecast is only trustworthy if its own track record is checked, not
  just its accuracy on a historical holdout at TRAINING time
  (benchmark.py's job). This module scores forecasts that were ACTUALLY
  SERVED, once the real outcome is known, broken down by lead time
  (horizon) and time of day (block) -- exactly where a forecast tends to
  be weakest -- against the SAME three untrained baselines
  (src.evaluation.baselines) used everywhere else in this project, so a
  "beats baseline" claim always means the same thing. Per the roadmap:
  "contains no number that cannot be traced to raw data" -- every figure
  below is a direct nMAE/nRMSE computed from the actual/predicted pairs
  given, never a fabricated or estimated value.

WHAT THIS DOES NOT DO:
  Fetch or store served forecasts itself -- this platform has no
  forecast log yet (POST /forecast doesn't persist what it returns; see
  README Known Gaps). This module scores whatever forecast-vs-actual
  pairs the caller supplies. "Generated unattended" here means the
  SCORING needs no human judgment calls (it's pure, deterministic
  arithmetic given its inputs) -- it does not mean this session built a
  cron job or scheduler; running this on a real weekly cadence is
  deployment infrastructure, not implemented here.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.evaluation.baselines import (
    clear_sky_power,
    nmae_nrmse,
    persistence_baseline,
    physics_baseline,
    smart_persistence_baseline,
)

METHOD_NAMES = ("model", "persistence", "smart_persistence", "physics")


def _score_group(df: pd.DataFrame, capacity_mw: float) -> dict:
    result = {}
    for name, col in [
        ("model", "model_mw"),
        ("persistence", "persistence_mw"),
        ("smart_persistence", "smart_persistence_mw"),
        ("physics", "physics_mw"),
    ]:
        nmae, nrmse, n = nmae_nrmse(df["actual_mw"], df[col], capacity_mw, df["daytime"])
        result[name] = {
            "nmae_pct": None if np.isnan(nmae) else round(float(nmae), 3),
            "nrmse_pct": None if np.isnan(nrmse) else round(float(nrmse), 3),
            "n": n,
        }
    return result


@dataclass
class WeeklyReport:
    plant_id: str
    period_start: object
    period_end: object
    n_forecasts: int
    by_horizon: dict
    by_block: dict
    overall: dict
    beats_baseline: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        def _ts(x):
            return x.isoformat() if hasattr(x, "isoformat") else x

        return {
            "plant_id": self.plant_id,
            "period_start": _ts(self.period_start),
            "period_end": _ts(self.period_end),
            "n_forecasts": self.n_forecasts,
            "overall": self.overall,
            "by_horizon_hours": {str(k): v for k, v in self.by_horizon.items()},
            "by_block_hour_of_day": {str(k): v for k, v in self.by_block.items()},
            "beats_baseline": self.beats_baseline,
        }


def generate_weekly_report(
    forecasts: pd.DataFrame,
    features: pd.DataFrame,
    plant_config: dict,
) -> WeeklyReport:
    """
    Score a batch of already-served forecasts against what actually
    happened, by horizon and by time-of-day block, against the three
    parameter-free baselines.

    Parameters
    ----------
    forecasts : pd.DataFrame
        One row per served forecast, columns:
          - target_timestamp: when the forecast was FOR (tz-aware)
          - horizon_hours: how far ahead it was made (e.g. 1, 6, 24)
          - predicted_mw: the model's own prediction for that block
    features : pd.DataFrame
        Built features (src.features.pipeline.build_features output, or
        equivalent) covering every `target_timestamp` above AND at least
        `src.evaluation.baselines.LAG_HOURS` hours before the EARLIEST
        one -- persistence/smart_persistence need that lookback to
        compute their own predictions. Must contain `solar_output_mw`
        (the actual, now-known outcome), `shortwave_radiation` and
        `clear_sky_ghi_model`.
    plant_config : dict
        See src.utils.config_loader.get_plant_config.

    Raises
    ------
    ValueError
        If `forecasts` is empty, or any `target_timestamp` isn't covered
        by `features` -- a report cannot honestly score a forecast whose
        outcome isn't in the data it was given.
    """
    if len(forecasts) == 0:
        raise ValueError("forecasts is empty -- nothing to report on.")

    capacity_mw = plant_config["capacity"]["ac_capacity_mw"]
    y = features["solar_output_mw"]
    daytime = features["clear_sky_ghi_model"] > 1.0

    persistence = persistence_baseline(y)
    smart_persistence = smart_persistence_baseline(y, features, plant_config)
    physics = physics_baseline(features, plant_config)

    targets = pd.DatetimeIndex(pd.to_datetime(forecasts["target_timestamp"]))
    missing = targets.difference(features.index)
    if len(missing) > 0:
        raise ValueError(
            f"{len(missing)} target_timestamp(s) in `forecasts` are not covered by `features` "
            f"(e.g. {missing[0]}) -- cannot score a forecast against an outcome that wasn't given."
        )

    scored = pd.DataFrame({
        "target_timestamp": targets,
        "horizon_hours": forecasts["horizon_hours"].to_numpy(),
        "actual_mw": y.reindex(targets).to_numpy(),
        "model_mw": forecasts["predicted_mw"].to_numpy(),
        "persistence_mw": persistence.reindex(targets).to_numpy(),
        "smart_persistence_mw": smart_persistence.reindex(targets).to_numpy(),
        "physics_mw": physics.reindex(targets).to_numpy(),
        "daytime": daytime.reindex(targets).fillna(False).to_numpy(),
    })

    by_horizon = {
        int(horizon): _score_group(group, capacity_mw)
        for horizon, group in scored.groupby("horizon_hours")
    }
    by_block = {
        int(hour): _score_group(group, capacity_mw)
        for hour, group in scored.groupby(scored["target_timestamp"].dt.hour)
    }
    overall = _score_group(scored, capacity_mw)

    beats_baseline = {}
    model_nmae = overall["model"]["nmae_pct"]
    if model_nmae is not None:
        for baseline in ("persistence", "smart_persistence", "physics"):
            baseline_nmae = overall[baseline]["nmae_pct"]
            beats_baseline[baseline] = (
                None if baseline_nmae is None else bool(model_nmae < baseline_nmae)
            )

    return WeeklyReport(
        plant_id=plant_config.get("plant_id", "unknown"),
        period_start=targets.min(),
        period_end=targets.max(),
        n_forecasts=len(scored),
        by_horizon=by_horizon,
        by_block=by_block,
        overall=overall,
        beats_baseline=beats_baseline,
    )
