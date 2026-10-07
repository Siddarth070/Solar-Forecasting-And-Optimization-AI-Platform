"""
attribution_validation.py — roadmap P3.4: validate loss attribution
against real O&M logs.
----------------------------------------------------------------------

WHAT THIS IS:
  Same "build the engine ahead of P3.1" pattern as
  src/evaluation/shadow_backtest.py (P3.2). `src/attribution/loss.py`
  (P2.4) already labels "equipment" and "curtailment" as SUSPECTED/
  POSSIBLE, not confirmed -- its own docstring says so, because this
  platform has no per-inverter telemetry and no grid curtailment-order
  feed. P3.4's whole point is checking those two suspected/possible
  labels against a real plant's own outage and curtailment log, and
  reporting precision/recall -- not guessing at whether the rule-based
  thresholds are any good.

  Nothing here runs until P3.1 lands real data; see
  tests/test_attribution_validation.py, which uses only synthetic
  fixtures and says so explicitly.

WHY ONLY "equipment" AND "curtailment":
  Those are the two causes `attribute_losses()` itself flags as
  unconfirmed. "weather" is cross-checked against measured irradiance
  already (that's what makes it "weather" rather than "unknown"), and an
  O&M log has no ground truth for weather anyway. "unknown" is already
  the module's own honest fallback for insufficient evidence -- scoring
  it against a log would just be asking "did the log confirm our
  shrug", which isn't a useful question.

EXPECTED O&M LOG SCHEMA:
  Exactly three columns -- `start_timestamp`, `end_timestamp`,
  `event_type` (one of "outage", "curtailment") -- the same "outage
  notes" this project's own P3.1 data request already asks a plant
  operator for. As with shadow_backtest.py's CSV loader, this is
  deliberately narrow: nobody has sent a real log yet, so there is
  nothing to guess a wider schema from.
"""

from pathlib import Path

import pandas as pd
from loguru import logger

from src.attribution.loss import CAUSE_CURTAILMENT, CAUSE_EQUIPMENT, LossAttributionReport

REQUIRED_COLUMNS = ("start_timestamp", "end_timestamp", "event_type")
VALIDATABLE_CAUSES = (CAUSE_EQUIPMENT, CAUSE_CURTAILMENT)
_EVENT_TYPE_TO_CAUSE = {"outage": CAUSE_EQUIPMENT, "curtailment": CAUSE_CURTAILMENT}


class AttributionValidationError(Exception):
    """Raised when the supplied O&M log, or the data it contains, can't
    be turned into ground truth to validate against."""


def load_om_log_csv(path: str | Path) -> pd.DataFrame:
    """Parse a plant-supplied O&M log CSV (see module docstring for the
    exact expected schema). Returns a DataFrame with a tz-aware
    `start_timestamp`/`end_timestamp` and a `cause` column already
    mapped from `event_type` -- rows with an event_type this module
    doesn't know how to map are dropped with a warning, not silently
    miscoded."""
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise AttributionValidationError(
            f"O&M log is missing required column(s) {missing} -- expected exactly "
            f"{REQUIRED_COLUMNS}, got {list(df.columns)}."
        )

    for col in ("start_timestamp", "end_timestamp"):
        try:
            parsed = pd.Index([pd.Timestamp(t) for t in df[col]])
        except (ValueError, TypeError) as e:
            raise AttributionValidationError(f"Could not parse '{col}': {e}") from e
        if not isinstance(parsed, pd.DatetimeIndex):
            raise AttributionValidationError(
                f"'{col}' could not be parsed into one consistent timezone -- e.g. some "
                f"rows carry a UTC offset and others don't."
            )
        if parsed.tz is None:
            raise AttributionValidationError(
                f"'{col}' has no UTC offset -- every timestamp this project compares against "
                f"must be tz-aware (e.g. 2024-06-01T10:00:00+05:30), or it can't be safely "
                f"compared against the (tz-aware) generation data."
            )
        df[col] = parsed

    unknown_types = set(df["event_type"]) - set(_EVENT_TYPE_TO_CAUSE)
    if unknown_types:
        logger.warning(
            f"Dropping {len(unknown_types)} unrecognised event_type value(s) {sorted(unknown_types)} "
            f"-- this module only maps {sorted(_EVENT_TYPE_TO_CAUSE)}."
        )
        df = df[df["event_type"].isin(_EVENT_TYPE_TO_CAUSE)]

    df["cause"] = df["event_type"].map(_EVENT_TYPE_TO_CAUSE)
    return df.reset_index(drop=True)


def ground_truth_cause_for_timestamp(ts, om_log: pd.DataFrame) -> str | None:
    """The O&M log's own cause for one block's timestamp, or None if no
    logged event covers it -- `None` means "the log can't confirm or
    deny this block", not "nothing happened"."""
    covering = om_log[(om_log["start_timestamp"] <= ts) & (ts <= om_log["end_timestamp"])]
    if covering.empty:
        return None
    # A block covered by more than one logged event is a real, if rare,
    # possibility (overlapping maintenance + curtailment windows) -- take
    # the first rather than silently picking one by sort order.
    return covering.iloc[0]["cause"]


def validate_attribution(report: LossAttributionReport, om_log: pd.DataFrame) -> dict:
    """Roadmap P3.4: compare `attribute_losses()`'s "equipment"/
    "curtailment" calls against a real O&M log, and report precision/
    recall per class -- the acceptance criterion in full. Causes below
    what the log can confirm are reported as such, not guessed past.
    """
    rows = []
    for block in report.blocks:
        gt = ground_truth_cause_for_timestamp(block.timestamp, om_log)
        rows.append({
            "timestamp": block.timestamp,
            "predicted_cause": block.cause,
            "confidence": block.confidence,
            "ground_truth_cause": gt,
        })
    comparison = pd.DataFrame(rows)

    n_total = len(comparison)
    n_scoreable = int(comparison["ground_truth_cause"].notna().sum())

    per_class = {}
    for cause in VALIDATABLE_CAUSES:
        scoreable = comparison[comparison["ground_truth_cause"].notna()]
        tp = int(((scoreable["predicted_cause"] == cause) & (scoreable["ground_truth_cause"] == cause)).sum())
        fp = int(((scoreable["predicted_cause"] == cause) & (scoreable["ground_truth_cause"] != cause)).sum())
        fn = int(((scoreable["predicted_cause"] != cause) & (scoreable["ground_truth_cause"] == cause)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f1 = (
            2 * precision * recall / (precision + recall)
            if (tp + fp) > 0 and (tp + fn) > 0 and (precision + recall) > 0
            else float("nan")
        )
        per_class[cause] = {
            "precision": precision, "recall": recall, "f1": f1,
            "true_positives": tp, "false_positives": fp, "false_negatives": fn,
            "n_logged_as_this_cause": int((scoreable["ground_truth_cause"] == cause).sum()),
        }

    return {
        "plant_id": report.plant_id,
        "n_blocks_total": n_total,
        "n_blocks_scoreable_against_log": n_scoreable,
        "n_blocks_unscoreable": n_total - n_scoreable,
        "per_class": per_class,
        "comparison": comparison,
    }


def validate_attribution_from_csv(report: LossAttributionReport, om_log_csv_path: str | Path) -> dict:
    """Convenience wrapper: load the O&M log from a plant-supplied CSV
    and run `validate_attribution` against it."""
    om_log = load_om_log_csv(om_log_csv_path)
    return validate_attribution(report, om_log)
