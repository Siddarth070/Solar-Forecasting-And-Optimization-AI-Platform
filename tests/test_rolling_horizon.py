"""
test_rolling_horizon.py
-------------------------
Roadmap P1.4: applying a proposed schedule revision under the real
CERC gate-closure timing (src/regulatory/grid_code.py) -- blocks before
the regulation's effective timestamp must stay locked at the OLD value
no matter what the new forecast says; only later blocks may change.

RUN WITH:
  pytest tests/test_rolling_horizon.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.scheduling.rolling_horizon import apply_schedule_revision
from src.time_blocks import block_boundaries

DATE = "2024-06-01"


def _full_day_series(value: float) -> pd.Series:
    edges = block_boundaries(DATE)[:-1]  # 96 block-start timestamps
    return pd.Series(np.full(96, value), index=edges)


class TestApplyScheduleRevision:
    def test_blocks_before_effective_timestamp_stay_locked(self):
        locked = _full_day_series(50.0)
        proposed = _full_day_series(70.0)
        request = pd.Timestamp("2024-06-01 13:56:00", tz="Asia/Kolkata")  # -> effective 15:30

        result = apply_schedule_revision(locked, proposed, request, transaction_type="bilateral")
        applied = result["applied_schedule"]

        before = applied[applied.index < result["effective_timestamp"]]
        after = applied[applied.index >= result["effective_timestamp"]]
        assert (before == 50.0).all()
        assert (after == 70.0).all()

    def test_effective_timestamp_matches_grid_code_module(self):
        locked = _full_day_series(50.0)
        proposed = _full_day_series(70.0)
        request = pd.Timestamp("2024-06-01 13:56:00", tz="Asia/Kolkata")

        result = apply_schedule_revision(locked, proposed, request, transaction_type="bilateral")
        assert result["effective_timestamp"] == pd.Timestamp("2024-06-01 15:30:00", tz="Asia/Kolkata")
        assert result["revision_allowed"] is True

    def test_locked_block_count_matches_the_boundary(self):
        locked = _full_day_series(50.0)
        proposed = _full_day_series(70.0)
        request = pd.Timestamp("2024-06-01 13:56:00", tz="Asia/Kolkata")

        result = apply_schedule_revision(locked, proposed, request, transaction_type="bilateral")
        # Effective timestamp 15:30 is block 63 (1-indexed) -> blocks 1-62 locked.
        assert result["locked_block_count"] == 62

    def test_collective_transaction_cannot_revise_at_all(self):
        locked = _full_day_series(50.0)
        proposed = _full_day_series(70.0)
        request = pd.Timestamp("2024-06-01 13:56:00", tz="Asia/Kolkata")

        result = apply_schedule_revision(locked, proposed, request, transaction_type="collective")
        assert result["revision_allowed"] is False
        assert result["effective_timestamp"] is None
        assert result["applied_schedule"].equals(locked)

    def test_mismatched_index_raises(self):
        locked = _full_day_series(50.0)
        proposed = _full_day_series(70.0).iloc[:-1]  # different length/index
        request = pd.Timestamp("2024-06-01 13:56:00", tz="Asia/Kolkata")

        with pytest.raises(ValueError):
            apply_schedule_revision(locked, proposed, request, transaction_type="bilateral")

    def test_realistic_forecast_update_only_changes_the_revisable_tail(self):
        """A more realistic scenario: locked schedule is yesterday's flat
        declaration; the proposed schedule reflects a genuinely different
        (non-flat) updated forecast. Only the tail from the effective
        timestamp onward should differ from the original locked values."""
        rng = np.random.default_rng(0)
        edges = block_boundaries(DATE)[:-1]
        locked = pd.Series(np.full(96, 40.0), index=edges)
        proposed = pd.Series(rng.uniform(0, 80, 96), index=edges)
        request = pd.Timestamp("2024-06-01 06:03:00", tz="Asia/Kolkata")

        result = apply_schedule_revision(locked, proposed, request, transaction_type="bilateral")
        applied = result["applied_schedule"]
        eff = result["effective_timestamp"]

        pd.testing.assert_series_equal(applied[applied.index < eff], locked[applied.index < eff])
        pd.testing.assert_series_equal(applied[applied.index >= eff], proposed[applied.index >= eff])
