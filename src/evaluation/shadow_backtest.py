"""
shadow_backtest.py — roadmap P3.2: retrospective shadow backtest.
--------------------------------------------------------------------

WHAT THIS IS:
  P3.1 (securing one real plant's historical dataset) is the hard
  blocker for the whole of Phase 3 -- every other P3.x task depends on
  it directly or transitively, and as of this writing it isn't done (no
  plant has agreed to share data; the outreach emails are drafted, not
  sent). This module is the P3.2 engine built AHEAD of that, so the
  moment a real CSV and an NDA exist, running the backtest is one
  function call, not a fresh build.

  It is deliberately NOT a reimplementation of benchmark.py's logic --
  it reuses the exact same P0.4 baselines and scoring convention
  (src.evaluation.baselines), the exact same feature pipeline
  (src.features.pipeline.build_features), the exact same data-quality
  gate (src.quality.gate, roadmap P2.2), and the exact same real
  weather source the live forecast path already uses
  (src.ingestion.open_meteo_fetcher.OpenMeteoFetcher.fetch_historical --
  hits the real Open-Meteo archive API, keyed only on the plant's own
  location, so the plant operator only has to hand over THEIR export
  readings, never weather data).

WHAT THIS DOES NOT DO:
  - Does not fabricate or simulate a real plant's history. The only
    accepted input is a CSV of genuinely real readings; every test in
    tests/test_shadow_backtest.py says so explicitly where it uses a
    synthetic fixture instead, and labels it as a dry run.
  - Does not guess at an unknown real-world CSV schema. The expected
    columns (`timestamp`, `power_mw`) are exactly what this project's
    own P3.1 data-request wording asks a plant operator for, so there
    is nothing to guess -- if a real file doesn't match, that's a
    conversation with that plant, not a parser to reverse-engineer.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.evaluation.baselines import (
    nmae_nrmse,
    persistence_baseline,
    physics_baseline,
    smart_persistence_baseline,
)
from src.features.pipeline import SERVING_FEATURE_COLUMNS, build_features
from src.ingestion.open_meteo_fetcher import OpenMeteoFetcher
from src.quality.gate import run_quality_checks
from src.utils.config_loader import get_plant_config

REQUIRED_COLUMNS = ("timestamp", "power_mw")


class ShadowBacktestError(Exception):
    """Raised when the supplied actuals file, or the data it contains,
    can't be turned into a scoreable series."""


def load_actual_readings_csv(path: str | Path) -> pd.Series:
    """Parse a plant-supplied CSV of real historical AC export readings.

    Expected format -- exactly the two columns this project's own P3.1
    data request asks a plant operator for:
        timestamp   ISO-8601, e.g. 2024-06-01T10:00:00+05:30
        power_mw    AC export in MW, >= 0

    Deliberately narrow: this project has never seen a real plant's own
    export file, so it only promises to parse the schema WE specify in
    the data-request, not every possible real-world quirk. Raises
    ShadowBacktestError (not a raw pandas exception) on anything else,
    same "name the problem, don't guess past it" convention as
    src/quality/gate.py.
    """
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ShadowBacktestError(
            f"CSV is missing required column(s) {missing} -- expected exactly "
            f"{REQUIRED_COLUMNS}, got {list(df.columns)}."
        )

    # Same "build one consistent DatetimeIndex or fail loudly" pattern as
    # src/api/main.py's build_timestamp_index -- a silent mix of tz-aware
    # and tz-naive timestamps is a real, common real-world export bug.
    try:
        idx = pd.Index([pd.Timestamp(t) for t in df["timestamp"]])
    except (ValueError, TypeError) as e:
        raise ShadowBacktestError(f"Could not parse the 'timestamp' column: {e}") from e
    if not isinstance(idx, pd.DatetimeIndex):
        raise ShadowBacktestError(
            "Timestamps could not be parsed into one consistent timezone -- e.g. "
            "some rows carry a UTC offset (+05:30) and others don't. This is a "
            "question for whoever exported the file, not something to guess past."
        )

    try:
        power = pd.to_numeric(df["power_mw"], errors="raise")
    except (ValueError, TypeError) as e:
        raise ShadowBacktestError(f"Could not parse the 'power_mw' column as numeric: {e}") from e

    series = pd.Series(power.to_numpy(dtype=float), index=idx, name="power_mw").sort_index()
    n_dup = int(series.index.duplicated().sum())
    if n_dup:
        logger.warning(f"{n_dup} duplicate timestamp(s) in actuals CSV -- keeping the first occurrence.")
        series = series[~series.index.duplicated(keep="first")]
    return series


def run_shadow_backtest(
    actual_power: pd.Series,
    plant_config: dict,
    served_model=None,
    meteo_fetcher: OpenMeteoFetcher | None = None,
) -> dict:
    """Roadmap P3.2: run Zenith's forecast over a real plant's history
    and compare against their actuals, with the P0.4 baselines shown
    alongside -- the acceptance criterion in full.

    Fetches REAL historical weather for this plant's own location over
    the exact date range `actual_power` covers (the same Open-Meteo
    archive source the live forecast path already uses), so the plant
    operator only has to supply their own export readings.

    `meteo_fetcher` is injectable so tests never hit the live network --
    production callers should leave it as None (a real OpenMeteoFetcher
    is constructed from `plant_config`).
    """
    if actual_power.empty:
        raise ShadowBacktestError("actual_power is empty -- nothing to backtest.")

    start_date = actual_power.index.min().strftime("%Y-%m-%d")
    end_date = actual_power.index.max().strftime("%Y-%m-%d")

    fetcher = meteo_fetcher if meteo_fetcher is not None else OpenMeteoFetcher(plant_config)
    weather = fetcher.fetch_historical(start_date, end_date)

    combined = weather.join(actual_power.rename("solar_output_mw"), how="left")
    features = build_features(combined, plant_config)

    y = features["solar_output_mw"]
    capacity_mw = plant_config["capacity"]["ac_capacity_mw"]
    daytime = features["clear_sky_ghi_model"] > 1.0

    persistence = persistence_baseline(y)
    smart_persistence = smart_persistence_baseline(y, features, plant_config)
    physics = physics_baseline(features, plant_config)

    scores: dict[str, dict] = {}
    for name, preds in [
        ("persistence", persistence),
        ("smart_persistence", smart_persistence),
        ("physics", physics),
    ]:
        nmae, nrmse, n = nmae_nrmse(y, preds, capacity_mw, daytime)
        scores[name] = {"nmae_pct": nmae, "nrmse_pct": nrmse, "n": n}

    if served_model is not None:
        # Score only rows where every serving feature is actually present --
        # filling a missing weather field with an invented value (e.g. 0)
        # would fabricate an input the model was never meant to see.
        valid = features[SERVING_FEATURE_COLUMNS].notna().all(axis=1)
        n_dropped = int((~valid).sum())
        if n_dropped:
            logger.warning(
                f"Dropping {n_dropped} row(s) with missing weather feature(s) "
                "before scoring the served model."
            )
        model_preds = pd.Series(
            np.clip(
                served_model.predict(features.loc[valid, SERVING_FEATURE_COLUMNS]) * capacity_mw,
                0, capacity_mw,
            ),
            index=features.index[valid],
        )
        nmae, nrmse, n = nmae_nrmse(y.loc[valid], model_preds, capacity_mw, daytime.loc[valid])
        scores["model"] = {"nmae_pct": nmae, "nrmse_pct": nrmse, "n": n}

    return {
        "plant_id": plant_config.get("plant_id", "unknown"),
        "period": {"start": start_date, "end": end_date},
        "n_readings": int(len(actual_power)),
        "scores": scores,
    }


def shadow_backtest_from_csv(
    csv_path: str | Path,
    plant_id: str,
    served_model=None,
    meteo_fetcher: OpenMeteoFetcher | None = None,
) -> dict:
    """End-to-end P3.2 entry point: load a real readings CSV, run the
    data-quality gate on it first (roadmap P2.2's own rule -- "the
    customer sees a quality report before they see a forecast"), and
    only backtest if it passes."""
    plant_config = get_plant_config(plant_id)
    actual_power = load_actual_readings_csv(csv_path)

    quality_report = run_quality_checks(
        actual_power.to_frame(name="power_mw"), plant_config, power_column="power_mw",
    )

    if not quality_report.passed:
        return {
            "plant_id": plant_id,
            "status": "blocked_on_data_quality",
            "quality_report": quality_report.to_dict(),
            "message": (
                "Per roadmap P2.2: the customer sees a quality report before "
                "they see a forecast. Resolve the error(s) above (or confirm "
                "they're expected and re-run) before backtesting."
            ),
        }

    result = run_shadow_backtest(
        actual_power, plant_config, served_model=served_model, meteo_fetcher=meteo_fetcher,
    )
    result["status"] = "ok"
    result["quality_report"] = quality_report.to_dict()
    return result
