"""
loss.py — Loss attribution engine (roadmap P2.4).

WHY THIS EXISTS:
  A forecast miss or a low-generation day is meaningless to an operator
  without a reason. This module compares actual generation against
  expected-from-irradiance (the same PV physics used to generate this
  project's own training data — see the WHAT THIS REUSES note below) and
  classifies each block's residual loss into a cause: weather, equipment,
  curtailment, soiling, or unknown. Per the roadmap: "Rule-based with
  explicit thresholds and a confidence label. Never outputs a cause
  without the evidence that produced it" — every classification below
  carries the numbers that produced it, not just a label.

WHAT THIS REUSES (never fabricated):
  `expected_generation_mw()` is exactly the physics formula
  src/ingestion/jaipur_simulator.py uses to generate its ground-truth
  solar_output_mw from GHI and temperature (P = GHI/1000 * temp_factor *
  performance_ratio * capacity), just parameterized by each plant's own
  `capacity.performance_ratio` / `capacity.temperature_coefficient` /
  `capacity.ac_capacity_mw` (configs/plants/<id>.yaml) instead of that
  simulator's hardcoded constants. The clear-sky index reuses
  `src.features.pipeline.clear_sky_ghi`, the same leak-free clear-sky
  model the forecast's own feature pipeline and the data-quality gate
  (src/quality/gate.py) already use.

WHAT THIS DOES NOT DO (real, documented gaps — see README Known Gaps):
  - "equipment" cannot be confirmed as a specific asset: this platform
    has no per-inverter/string telemetry, only one plant-level aggregate
    power reading. A sustained, irradiance-uncorrelated drop is labelled
    "equipment (suspected)", never a specific inverter.
  - "curtailment" cannot be confirmed against a real grid instruction:
    this platform has no SLDC/RLDC curtailment-order feed (that needs a
    real QCA/grid operator integration — roadmap P2.7/P2.8, both
    explicitly deferred). A hard flat clip is labelled "curtailment
    (possible)", never a confirmed grid-ordered curtailment.
  - "soiling" needs a multi-day trend, not a single block — see
    `daily_performance_ratio()` / `detect_soiling()` below, which operate
    on a day-level series rather than per-block.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.features.pipeline import clear_sky_ghi

# Ignore a loss smaller than this in absolute MW -- sensor/rounding noise,
# not a real loss worth attributing.
DEFAULT_MATERIALITY_LOSS_MW = 1.0
# ...or smaller than this as a fraction of that block's own expected
# output -- a 0.5 MW loss on a 1 MW-expected block is huge; on a 90 MW
# one it's nothing.
DEFAULT_MATERIALITY_LOSS_FRACTION = 0.05
# Skip classification entirely when expected output itself is below this
# fraction of AC capacity (dawn/dusk) -- expected is too close to zero for
# a loss fraction to mean anything.
DEFAULT_MIN_EXPECTED_MW_FRACTION = 0.05
# A clear-sky-index change of at least this much from the previous block
# reads as a passing cloud (a real, fast irradiance transient).
DEFAULT_WEATHER_RAMP_THRESHOLD = 0.15
# A run of material-loss blocks at least this long (1 hour at 15-min
# resolution), NOT explained by a cloud ramp, reads as a persistent fault
# rather than momentary noise.
DEFAULT_EQUIPMENT_MIN_RUN_BLOCKS = 4
# Within such a run, actual output varying less than this (MW, population
# std) reads as "pinned flat" rather than "reduced but still tracking
# irradiance".
DEFAULT_CURTAILMENT_FLAT_STD_MW = 1.0
# ...and expected output must vary at least this many times more than
# actual within that same run for the flatness to be meaningful (both
# being low, e.g. a uniformly overcast hour, is not curtailment).
DEFAULT_CURTAILMENT_VARIANCE_RATIO = 3.0

# Soiling: a sustained, gradual decline in the daily actual/expected ratio,
# distinct from a single-day step (equipment) or a same-day cloud dip
# (weather). Needs at least this many days of data to fit a trend on.
DEFAULT_SOILING_MIN_DAYS = 10
# A trend at least this steep (percentage points of ratio per day,
# negative) over that window is physically consistent with dust
# accumulation rather than daily noise.
DEFAULT_SOILING_SLOPE_THRESHOLD_PCT_PER_DAY = -0.1

CAUSE_WEATHER = "weather"
CAUSE_EQUIPMENT = "equipment"
CAUSE_CURTAILMENT = "curtailment"
CAUSE_UNKNOWN = "unknown"


def expected_generation_mw(df: pd.DataFrame, plant_config: dict,
                            ghi_column: str = "shortwave_radiation",
                            temperature_column: str = "temperature_2m") -> pd.Series:
    """Physics-based expected output from ACTUAL measured irradiance and
    temperature (not the clear-sky model -- that's a separate, weather-
    only baseline used only for the clear-sky-index ramp check below)."""
    capacity = plant_config["capacity"]
    ac_capacity_mw = capacity["ac_capacity_mw"]
    performance_ratio = capacity["performance_ratio"]
    temp_coefficient = capacity["temperature_coefficient"]

    temp_factor = 1 + temp_coefficient * (df[temperature_column] - 25)
    raw = df[ghi_column] / 1000 * temp_factor * performance_ratio
    expected = (raw * ac_capacity_mw).clip(lower=0, upper=ac_capacity_mw)
    return expected


def _clear_sky_index(df: pd.DataFrame, plant_config: dict,
                      ghi_column: str = "shortwave_radiation") -> pd.Series:
    location = plant_config["location"]
    csghi = clear_sky_ghi(df.index, location["latitude"], location["longitude"],
                           location.get("elevation_m", 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = df[ghi_column].to_numpy() / csghi.to_numpy()
    idx = np.where(csghi.to_numpy() > 1.0, raw, 0.0)
    return pd.Series(np.clip(idx, 0, 1.3), index=df.index)


@dataclass
class LossBlock:
    timestamp: object
    actual_mw: float
    expected_mw: float
    loss_mw: float
    loss_fraction: float
    cause: str
    confidence: str  # "low" | "medium" | "high"
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat() if hasattr(self.timestamp, "isoformat") else str(self.timestamp),
            "actual_mw": round(self.actual_mw, 3),
            "expected_mw": round(self.expected_mw, 3),
            "loss_mw": round(self.loss_mw, 3),
            "loss_fraction": round(self.loss_fraction, 4),
            "cause": self.cause,
            "confidence": self.confidence,
            "evidence": self.evidence,
        }


@dataclass
class LossAttributionReport:
    plant_id: str
    total_rows: int
    blocks: list

    @property
    def total_loss_mwh(self) -> float:
        """Approximate, assuming the caller's own block spacing -- callers
        needing a precise MWh figure should integrate with
        src.time_blocks.integrate_to_blocks instead; this is a quick
        diagnostic total, not a settlement number."""
        return sum(b.loss_mw for b in self.blocks)

    def causes_summary(self) -> dict:
        summary = {}
        for b in self.blocks:
            summary.setdefault(b.cause, {"count": 0, "loss_mw": 0.0})
            summary[b.cause]["count"] += 1
            summary[b.cause]["loss_mw"] += b.loss_mw
        for cause in summary:
            summary[cause]["loss_mw"] = round(summary[cause]["loss_mw"], 3)
        return summary

    def to_dict(self) -> dict:
        return {
            "plant_id": self.plant_id,
            "total_rows": self.total_rows,
            "material_loss_blocks": len(self.blocks),
            "causes_summary": self.causes_summary(),
            "blocks": [b.to_dict() for b in self.blocks],
        }


def attribute_losses(
    df: pd.DataFrame,
    plant_config: dict,
    power_column: str = "solar_output_mw",
    ghi_column: str = "shortwave_radiation",
    temperature_column: str = "temperature_2m",
    materiality_loss_mw: float = DEFAULT_MATERIALITY_LOSS_MW,
    materiality_loss_fraction: float = DEFAULT_MATERIALITY_LOSS_FRACTION,
    min_expected_mw_fraction: float = DEFAULT_MIN_EXPECTED_MW_FRACTION,
    weather_ramp_threshold: float = DEFAULT_WEATHER_RAMP_THRESHOLD,
    equipment_min_run_blocks: int = DEFAULT_EQUIPMENT_MIN_RUN_BLOCKS,
    curtailment_flat_std_mw: float = DEFAULT_CURTAILMENT_FLAT_STD_MW,
    curtailment_variance_ratio: float = DEFAULT_CURTAILMENT_VARIANCE_RATIO,
) -> LossAttributionReport:
    """
    Classify every block with a material generation shortfall against
    expected-from-irradiance into a cause, with the evidence that produced
    it. See the module docstring for what each cause does and does not
    confirm.

    Parameters
    ----------
    df : pd.DataFrame
        Indexed by timestamp, with `power_column` (actual MW),
        `ghi_column` (measured irradiance, W/m^2) and `temperature_column`
        (measured ambient temperature, degC).
    plant_config : dict
        See src.utils.config_loader.get_plant_config.
    """
    ac_capacity_mw = plant_config["capacity"]["ac_capacity_mw"]

    expected = expected_generation_mw(df, plant_config, ghi_column, temperature_column)
    actual = df[power_column]
    loss = (expected - actual).clip(lower=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        loss_fraction = np.where(expected.to_numpy() > 1e-6, loss.to_numpy() / expected.to_numpy(), 0.0)
    loss_fraction = pd.Series(loss_fraction, index=df.index)

    csi = _clear_sky_index(df, plant_config, ghi_column)
    csi_ramp = csi.diff().abs().fillna(0.0)

    min_expected_floor = min_expected_mw_fraction * ac_capacity_mw
    is_material = (
        (loss >= materiality_loss_mw)
        & (loss_fraction >= materiality_loss_fraction)
        & (expected >= min_expected_floor)
    )

    # Group consecutive material-loss blocks into runs: a new run_id starts
    # every time is_material CHANGES value (False->True or True->False), so
    # each run_id spans a homogeneous stretch -- unlike grouping only on
    # "~is_material", this doesn't leak the non-material block immediately
    # before a run into that run's own length.
    run_id = (is_material != is_material.shift()).cumsum()
    run_lengths = is_material.groupby(run_id).transform("size")

    blocks = []
    for i, ts in enumerate(df.index):
        if not bool(is_material.iloc[i]):
            continue

        run_length = int(run_lengths.iloc[i])
        ramp = float(csi_ramp.iloc[i])
        evidence = {
            "clear_sky_index": round(float(csi.iloc[i]), 3),
            "clear_sky_index_ramp": round(ramp, 3),
            "run_length_blocks": run_length,
        }

        if ramp >= weather_ramp_threshold:
            cause, confidence = CAUSE_WEATHER, "medium"
            evidence["reason"] = (
                f"clear-sky index changed by {ramp:.2f} from the previous block -- "
                f"consistent with a passing cloud."
            )
        elif run_length >= equipment_min_run_blocks:
            run_mask = (run_id == run_id.iloc[i])
            actual_run = actual[run_mask]
            expected_run = expected[run_mask]
            actual_std = float(actual_run.std(ddof=0))
            expected_std = float(expected_run.std(ddof=0))
            evidence["run_actual_std_mw"] = round(actual_std, 3)
            evidence["run_expected_std_mw"] = round(expected_std, 3)
            # The run's own start timestamp -- lets a downstream consumer
            # (e.g. src.recommendations, roadmap P2.6) group every block
            # of one physical event together, instead of re-deriving
            # adjacency from timestamps it wasn't given.
            run_start_timestamp = actual_run.index.min()
            evidence["run_start_timestamp"] = (
                run_start_timestamp.isoformat() if hasattr(run_start_timestamp, "isoformat")
                else str(run_start_timestamp)
            )

            if actual_std <= curtailment_flat_std_mw and expected_std >= actual_std * curtailment_variance_ratio:
                cause, confidence = CAUSE_CURTAILMENT, "medium"
                evidence["reason"] = (
                    f"output held flat (std {actual_std:.2f} MW) for {run_length} consecutive "
                    f"blocks while expected output varied {expected_std:.2f} MW -- looks like a hard "
                    f"output cap, not weather or a gradual fault. This platform has no real grid "
                    f"curtailment-instruction feed, so this is POSSIBLE curtailment, not confirmed."
                )
            else:
                cause, confidence = CAUSE_EQUIPMENT, "medium"
                evidence["reason"] = (
                    f"output stayed below expected for {run_length} consecutive blocks without a "
                    f"cloud-driven clear-sky-index ramp -- looks like a hardware fault. This platform "
                    f"has no per-inverter/string telemetry, so this is SUSPECTED equipment loss, not "
                    f"a confirmed asset."
                )
        else:
            cause, confidence = CAUSE_UNKNOWN, "low"
            evidence["reason"] = (
                "an isolated block of material loss, not explained by a cloud ramp and too short a "
                "run to read as a persistent fault -- insufficient evidence to attribute confidently."
            )

        blocks.append(LossBlock(
            timestamp=ts,
            actual_mw=float(actual.iloc[i]),
            expected_mw=float(expected.iloc[i]),
            loss_mw=float(loss.iloc[i]),
            loss_fraction=float(loss_fraction.iloc[i]),
            cause=cause,
            confidence=confidence,
            evidence=evidence,
        ))

    return LossAttributionReport(
        plant_id=plant_config.get("plant_id", "unknown"),
        total_rows=len(df),
        blocks=blocks,
    )


def daily_performance_ratio(
    df: pd.DataFrame,
    plant_config: dict,
    power_column: str = "solar_output_mw",
    ghi_column: str = "shortwave_radiation",
    temperature_column: str = "temperature_2m",
    min_expected_mw_fraction: float = DEFAULT_MIN_EXPECTED_MW_FRACTION,
) -> pd.Series:
    """
    Mean actual/expected ratio per calendar day, over only the daytime
    blocks (expected output at or above the noise floor) -- the day-level
    signal soiling needs, since any single block's ratio is too noisy
    (dawn/dusk, a passing cloud) to trend on.

    Returns a pd.Series indexed by calendar date (one value per day that
    had at least one qualifying daytime block).
    """
    ac_capacity_mw = plant_config["capacity"]["ac_capacity_mw"]
    expected = expected_generation_mw(df, plant_config, ghi_column, temperature_column)
    actual = df[power_column]
    floor = min_expected_mw_fraction * ac_capacity_mw

    daytime = expected >= floor
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(daytime.to_numpy(), actual.to_numpy() / expected.to_numpy(), np.nan)
    ratio = pd.Series(ratio, index=df.index)

    daily = ratio.groupby(df.index.date).mean()
    daily.index = pd.to_datetime(daily.index)
    return daily.dropna().sort_index()


@dataclass
class SoilingFinding:
    start_date: object
    end_date: object
    days_observed: int
    slope_pct_per_day: float
    confidence: str
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "start_date": self.start_date.isoformat() if hasattr(self.start_date, "isoformat") else str(self.start_date),
            "end_date": self.end_date.isoformat() if hasattr(self.end_date, "isoformat") else str(self.end_date),
            "days_observed": self.days_observed,
            "slope_pct_per_day": self.slope_pct_per_day,
            "cause": "soiling",
            "confidence": self.confidence,
            "evidence": self.evidence,
        }


def detect_soiling(
    daily_ratio: pd.Series,
    min_days: int = DEFAULT_SOILING_MIN_DAYS,
    slope_threshold_pct_per_day: float = DEFAULT_SOILING_SLOPE_THRESHOLD_PCT_PER_DAY,
):
    """
    A sustained, gradual decline in the daily actual/expected ratio over
    at least `min_days` days -- distinct from a single-day step
    (equipment, see `attribute_losses`) or a same-day cloud dip (weather).
    Fits one linear trend (least squares) across the whole window; only a
    material, steady decline this slow over this many days is physically
    consistent with dust/dirt accumulation rather than a discrete fault
    or ordinary day-to-day noise.

    Returns None when there isn't enough data, or the decline (if any)
    doesn't clear the threshold.
    """
    if len(daily_ratio) < min_days:
        return None

    x = np.arange(len(daily_ratio))
    y = daily_ratio.to_numpy()
    slope, _intercept = np.polyfit(x, y, 1)
    slope_pct_per_day = float(slope * 100)

    if slope_pct_per_day > slope_threshold_pct_per_day:
        return None

    return SoilingFinding(
        start_date=daily_ratio.index[0],
        end_date=daily_ratio.index[-1],
        days_observed=len(daily_ratio),
        slope_pct_per_day=round(slope_pct_per_day, 4),
        confidence="medium",
        evidence={
            "ratio_start": round(float(y[0]), 4),
            "ratio_end": round(float(y[-1]), 4),
            "reason": (
                f"daily actual/expected ratio declined by {abs(slope_pct_per_day):.3f} percentage "
                f"points/day over {len(daily_ratio)} days ({y[0]:.3f} -> {y[-1]:.3f}) -- consistent "
                f"with gradual soiling rather than a discrete equipment fault or a single cloudy day."
            ),
        },
    )
