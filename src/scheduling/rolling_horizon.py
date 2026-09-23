"""
rolling_horizon.py — Applying a schedule revision under real gate-closure
timing (roadmap P1.4).

WHY THIS EXISTS:
  A day-ahead schedule is not something a plant can silently overwrite
  intraday the moment a better forecast arrives. Under CERC's Grid Code
  (src/regulatory/grid_code.py), any revision only takes effect several
  blocks after it is requested (Regulation 49(4)(c)'s 7th/8th time block
  rule), and a WS seller can only revise at all if it sells under a
  bilateral transaction structure (Regulation 49(8)). This module applies
  that real constraint to a proposed schedule update: the near-term
  blocks between "now" and the regulation's effective timestamp stay
  LOCKED at whatever was already declared, no matter how much better the
  new forecast is; only blocks from the effective timestamp onward can
  actually change.

THIS IS NOT OPTIMIZATION:
  This module doesn't decide what a good revised schedule looks like --
  that's src/models (the forecast) and src/optimization (the battery LP).
  It only decides WHICH blocks of a proposed revision are allowed to take
  effect and when, per the real regulatory timing.
"""

import pandas as pd

from src.regulatory.grid_code import can_revise_schedule, revision_effective_timestamp


def apply_schedule_revision(
    locked_schedule: pd.Series,
    proposed_schedule: pd.Series,
    request_timestamp,
    transaction_type: str = "bilateral",
) -> dict:
    """
    Merge a proposed schedule revision into the currently-locked schedule,
    respecting the real gate-closure timing.

    Parameters
    ----------
    locked_schedule : pd.Series
        The schedule currently in effect (e.g. yesterday's D-1 declaration),
        indexed by block-start timestamps (see src/time_blocks.py).
    proposed_schedule : pd.Series
        A candidate revised schedule (e.g. from an updated intraday
        forecast), indexed the SAME way as `locked_schedule`.
    request_timestamp :
        When the revision is being requested (real time "now").
    transaction_type : str
        "bilateral" or "collective" (Regulation 49(8)) -- only bilateral
        WS-seller transactions may revise at all.

    Returns
    -------
    dict with:
      "applied_schedule": pd.Series -- locked_schedule for every block
          before the regulation's effective timestamp, proposed_schedule
          from that timestamp onward.
      "effective_timestamp": pd.Timestamp or None (None if the revision
          isn't permitted at all -- see `revision_allowed`).
      "revision_allowed": bool
      "locked_block_count": int -- how many blocks of `proposed_schedule`
          were rejected (still locked) because they fall before the
          effective timestamp.
    """
    if not locked_schedule.index.equals(proposed_schedule.index):
        raise ValueError("locked_schedule and proposed_schedule must share the same index")

    if not can_revise_schedule(transaction_type):
        return {
            "applied_schedule": locked_schedule.copy(),
            "effective_timestamp": None,
            "revision_allowed": False,
            "locked_block_count": len(locked_schedule),
        }

    effective_ts = revision_effective_timestamp(request_timestamp)
    is_locked = locked_schedule.index < effective_ts
    applied = locked_schedule.where(is_locked, proposed_schedule)

    return {
        "applied_schedule": applied,
        "effective_timestamp": effective_ts,
        "revision_allowed": True,
        "locked_block_count": int(is_locked.sum()),
    }
