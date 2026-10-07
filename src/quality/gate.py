"""
gate.py — Data-quality gate for ingested generation data (roadmap P2.2).

WHY THIS EXISTS:
  Real customer files have real problems: gaps where the logger dropped
  out, duplicate rows from a re-export, a meter that reports through the
  night, an inverter stuck reporting yesterday's last value, values that
  exceed the plant's own rated capacity. Feeding any of this straight
  into a forecast silently corrupts it. Per the roadmap: "The customer
  sees a quality report before they see a forecast. This builds more
  trust than the forecast does" -- so this module's job is to find and
  NAME these problems, not to fix them silently.

WHAT THIS DOES NOT DO:
  Detect a mislabeled timezone by inference (e.g. data secretly in UTC
  claiming to be IST) -- that needs a real ground truth to compare
  against. The `night_time_non_zero` check catches the specific,
  extremely common case of a ~5:30h offset error (UTC vs IST) as a side
  effect, since real solar position at the CLAIMED timestamp would show
  the sun below the horizon while the data reports generation -- but a
  tz-aware, wrongly-labeled-by-a-smaller-offset error would not
  necessarily trip it.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.features.pipeline import clear_sky_ghi

DEFAULT_MIN_FLATLINE_BLOCKS = 4  # 1 hour at 15-min resolution, or 4 rows at any resolution
DEFAULT_CAPACITY_TOLERANCE = 1.02  # 2% headroom for meter/measurement noise
DEFAULT_NEGATIVE_TOLERANCE_MW = -0.01  # floating-point noise around zero, not a real negative
DEFAULT_ZERO_THRESHOLD_MW = 0.01  # readings at or below this count as "zero" (nighttime), not a flatline fault
DEFAULT_NIGHT_TIME_THRESHOLD_MW = 0.5  # generation above this when the sun is down is a real fault
DEFAULT_CLEAR_SKY_NIGHT_GHI = 1.0  # W/m^2 -- below this, pvlib's Ineichen model says "night"


@dataclass
class QualityIssue:
    check: str
    severity: str  # "error" (data is unsafe to forecast on) or "warning" (usable, but noted)
    count: int
    timestamps: list = field(default_factory=list)
    message: str = ""

    def to_dict(self) -> dict:
        return {
            "check": self.check,
            "severity": self.severity,
            "count": self.count,
            "timestamps": [t.isoformat() if hasattr(t, "isoformat") else str(t) for t in self.timestamps],
            "message": self.message,
        }


@dataclass
class QualityReport:
    plant_id: str
    total_rows: int
    issues: list

    @property
    def passed(self) -> bool:
        """No ERROR-severity issue -- warnings alone don't fail the gate."""
        return not any(i.severity == "error" for i in self.issues)

    @property
    def error_count(self) -> int:
        return sum(i.count for i in self.issues if i.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(i.count for i in self.issues if i.severity == "warning")

    def to_dict(self) -> dict:
        return {
            "plant_id": self.plant_id,
            "total_rows": self.total_rows,
            "passed": self.passed,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "issues": [i.to_dict() for i in self.issues],
        }

    def summary(self) -> str:
        if not self.issues:
            return f"[{self.plant_id}] {self.total_rows} rows -- no quality issues found."
        lines = [f"[{self.plant_id}] {self.total_rows} rows -- "
                 f"{'PASSED' if self.passed else 'FAILED'} "
                 f"({self.error_count} error rows, {self.warning_count} warning rows)"]
        for issue in self.issues:
            lines.append(f"  [{issue.severity.upper()}] {issue.check}: {issue.message}")
        return "\n".join(lines)


_NOT_A_DATETIME_INDEX = object()  # sentinel: df.index isn't a real DatetimeIndex at all


def _check_missing_timezone(df: pd.DataFrame) -> QualityIssue | None:
    """Flags two distinct real-world cases as the same error: every
    timestamp is naive (df.index.tz is None, a proper DatetimeIndex), or
    the batch MIXES tz-aware and tz-naive timestamps -- e.g. one bad row
    from a different export -- which pandas can't even unify into a
    single DatetimeIndex, so df.index falls back to a generic object
    Index with no `.tz` attribute at all. Either way, this data cannot be
    safely aligned to the 96-block IST grid or checked against real
    sunrise/sunset, so both read as the same quality failure."""
    tz = getattr(df.index, "tz", _NOT_A_DATETIME_INDEX)
    if tz is _NOT_A_DATETIME_INDEX:
        return QualityIssue(
            check="missing_timezone", severity="error", count=len(df),
            message="Timestamps could not be parsed into one consistent timezone -- e.g. "
                    "some rows carry a UTC offset and others don't. Cannot be safely "
                    "aligned to the 96-block IST grid or checked against real "
                    "sunrise/sunset.",
        )
    if tz is None:
        return QualityIssue(
            check="missing_timezone", severity="error", count=len(df),
            message="Timestamps have no timezone -- cannot be safely aligned to the "
                    "96-block IST grid or checked against real sunrise/sunset.",
        )
    return None


def _check_duplicate_timestamps(df: pd.DataFrame) -> QualityIssue | None:
    dupes = df.index[df.index.duplicated(keep=False)]
    if len(dupes) == 0:
        return None
    return QualityIssue(
        check="duplicate_timestamps", severity="error", count=len(dupes),
        timestamps=sorted(set(dupes)),
        message=f"{len(set(dupes))} distinct timestamp(s) appear more than once "
                f"({len(dupes)} rows total).",
    )


def _check_timestamp_gaps(df: pd.DataFrame) -> QualityIssue | None:
    """Infer the nominal resolution as the MODE of consecutive gaps (robust
    to a handful of genuine gaps skewing a mean/min), then flag any
    interval that is a multiple of that resolution larger than expected."""
    idx = df.index.sort_values()
    if len(idx) < 3:
        return None
    deltas = idx.to_series().diff().dropna()
    nominal = deltas.mode().iloc[0]
    if nominal <= pd.Timedelta(0):
        return None
    gap_mask = deltas > nominal * 1.5
    if not gap_mask.any():
        return None
    gap_starts = deltas.index[gap_mask]
    missing_blocks = int(((deltas[gap_mask] / nominal).round() - 1).sum())
    return QualityIssue(
        check="timestamp_gaps", severity="warning", count=missing_blocks,
        timestamps=list(gap_starts),
        message=f"{len(gap_starts)} gap(s) totalling ~{missing_blocks} missing "
                f"block(s) at the inferred {nominal} resolution.",
    )


def _check_negative_power(df: pd.DataFrame, power_column: str) -> QualityIssue | None:
    mask = df[power_column] < DEFAULT_NEGATIVE_TOLERANCE_MW
    if not mask.any():
        return None
    return QualityIssue(
        check="negative_power", severity="error", count=int(mask.sum()),
        timestamps=list(df.index[mask]),
        message=f"{int(mask.sum())} block(s) report negative generation "
                f"(most negative: {df.loc[mask, power_column].min():.3f} MW).",
    )


def _check_above_capacity(df: pd.DataFrame, power_column: str, ac_capacity_mw: float) -> QualityIssue | None:
    ceiling = ac_capacity_mw * DEFAULT_CAPACITY_TOLERANCE
    mask = df[power_column] > ceiling
    if not mask.any():
        return None
    return QualityIssue(
        check="above_capacity", severity="error", count=int(mask.sum()),
        timestamps=list(df.index[mask]),
        message=f"{int(mask.sum())} block(s) exceed {ceiling:.1f} MW "
                f"({DEFAULT_CAPACITY_TOLERANCE:.0%} of {ac_capacity_mw:.0f} MW AC capacity) "
                f"(max reported: {df.loc[mask, power_column].max():.1f} MW).",
    )


def _check_flatlines(df: pd.DataFrame, power_column: str,
                      min_flatline_blocks: int = DEFAULT_MIN_FLATLINE_BLOCKS) -> QualityIssue | None:
    """A stuck sensor repeats the exact same value; a run of >= N identical
    NON-ZERO consecutive readings (zero at night is normal, not a fault)."""
    values = df[power_column].to_numpy()
    is_repeat = np.concatenate([[False], np.isclose(values[1:], values[:-1], atol=1e-6)])
    is_nonzero = values > DEFAULT_ZERO_THRESHOLD_MW
    run_id = (~(is_repeat & is_nonzero)).cumsum()
    run_lengths = pd.Series(1, index=df.index).groupby(run_id).transform("count")
    flatline_mask = (is_repeat & is_nonzero) & (run_lengths >= min_flatline_blocks)
    # Include each run's first block too (the one is_repeat itself skips).
    flatline_mask = flatline_mask | flatline_mask.shift(-1, fill_value=False) & is_nonzero & ~is_repeat
    if not flatline_mask.any():
        return None
    return QualityIssue(
        check="flatline", severity="warning", count=int(flatline_mask.sum()),
        timestamps=list(df.index[flatline_mask]),
        message=f"{int(flatline_mask.sum())} block(s) show a non-zero reading repeated "
                f"for at least {min_flatline_blocks} consecutive blocks (possible stuck inverter).",
    )


def _check_night_time_non_zero(df: pd.DataFrame, power_column: str, plant_config: dict) -> QualityIssue | None:
    location = plant_config["location"]
    csghi = clear_sky_ghi(df.index, location["latitude"], location["longitude"],
                           location.get("elevation_m", 0.0))
    is_night = csghi.to_numpy() < DEFAULT_CLEAR_SKY_NIGHT_GHI
    mask = is_night & (df[power_column].to_numpy() > DEFAULT_NIGHT_TIME_THRESHOLD_MW)
    if not mask.any():
        return None
    return QualityIssue(
        check="night_time_non_zero", severity="error", count=int(mask.sum()),
        timestamps=list(df.index[mask]),
        message=f"{int(mask.sum())} block(s) report generation above "
                f"{DEFAULT_NIGHT_TIME_THRESHOLD_MW} MW while the sun is below the horizon "
                f"at this plant's location -- check for a timezone or meter fault.",
    )


def run_quality_checks(df: pd.DataFrame, plant_config: dict,
                        power_column: str = "solar_output_mw") -> QualityReport:
    """
    Run every data-quality check against one plant's ingested generation
    data and return a single report.

    Parameters
    ----------
    df : pd.DataFrame
        Indexed by timestamp (tz-aware, ideally), with a `power_column`
        of generation readings in MW.
    plant_config : dict
        This plant's config (see src.utils.config_loader.get_plant_config)
        -- location for the night-time check, capacity for the
        above-capacity check.
    power_column : str
        Which column of `df` holds the generation reading.
    """
    issues = []

    tz_issue = _check_missing_timezone(df)
    if tz_issue:
        issues.append(tz_issue)

    dup_issue = _check_duplicate_timestamps(df)  # safe on any index dtype
    if dup_issue:
        issues.append(dup_issue)

    # Gap detection sorts the index -- a mix of tz-aware and tz-naive
    # timestamps can't be sorted at all (pandas raises), so skip it
    # (rather than crash) if the timezone check already failed.
    if tz_issue is None:
        gap_issue = _check_timestamp_gaps(df)
        if gap_issue:
            issues.append(gap_issue)

    for issue in (
        _check_negative_power(df, power_column),
        _check_above_capacity(df, power_column, plant_config["capacity"]["ac_capacity_mw"]),
        _check_flatlines(df, power_column),
    ):
        if issue:
            issues.append(issue)

    # Night-time check needs a real tz-aware index for clear_sky_ghi --
    # skip it (rather than crash) if the timezone check already failed.
    if tz_issue is None:
        night_issue = _check_night_time_non_zero(df, power_column, plant_config)
        if night_issue:
            issues.append(night_issue)

    return QualityReport(
        plant_id=plant_config.get("plant_id", "unknown"),
        total_rows=len(df),
        issues=issues,
    )
