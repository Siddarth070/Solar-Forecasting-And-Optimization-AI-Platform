"""
test_grid_code.py
-------------------
Roadmap P1.4 acceptance criterion: rolling day-ahead/intraday horizons
respecting the REAL CERC Indian Electricity Grid Code (IEGC) 2023
revision-timing rules -- tested directly against the regulation text and
CERC's own removal-of-difficulties orders (see src/regulatory/grid_code.py's
module docstring for full citations).

RUN WITH:
  pytest tests/test_grid_code.py -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.regulatory import grid_code
from src.time_blocks import block_of


class TestRevisionEffectiveTimestamp:
    """Regulation 49(4)(c): revision in an ODD block -> effective at the
    7th block (k+6); revision in an EVEN block -> effective at the 8th
    block (k+7), counting the request's own block as the first."""

    def test_matches_cerc_orders_own_worked_example(self):
        """14/SM/2023, para 14: a request at 1.56 PM (block 56,
        13:45-14:00, EVEN) becomes effective at 3.30-3.45 PM (block 63) --
        the order's own arithmetic, not just this module's."""
        request = pd.Timestamp("2024-06-01 13:56:00", tz="Asia/Kolkata")
        assert block_of(request) == 56
        effective = grid_code.revision_effective_timestamp(request)
        assert effective == pd.Timestamp("2024-06-01 15:30:00", tz="Asia/Kolkata")

    def test_odd_block_request_becomes_effective_6_blocks_later(self):
        # Block 55 spans 13:30-13:45 (odd) -> effective at block 61
        # (55+6), which spans 15:00-15:15.
        request = pd.Timestamp("2024-06-01 13:35:00", tz="Asia/Kolkata")
        assert block_of(request) == 55
        effective = grid_code.revision_effective_timestamp(request)
        assert effective == pd.Timestamp("2024-06-01 15:00:00", tz="Asia/Kolkata")

    def test_even_block_request_becomes_effective_7_blocks_later(self):
        # Block 2 spans 00:15-00:30 (even) -> effective at block 9 (2+7),
        # which spans 02:00-02:15.
        request = pd.Timestamp("2024-06-01 00:20:00", tz="Asia/Kolkata")
        assert block_of(request) == 2
        effective = grid_code.revision_effective_timestamp(request)
        assert effective == pd.Timestamp("2024-06-01 02:00:00", tz="Asia/Kolkata")

    def test_first_block_of_day_is_odd(self):
        # Block 1 spans 00:00-00:15 (odd) -> effective at block 7 (1+6),
        # which spans 01:30-01:45.
        request = pd.Timestamp("2024-06-01 00:05:00", tz="Asia/Kolkata")
        assert block_of(request) == 1
        effective = grid_code.revision_effective_timestamp(request)
        assert effective == pd.Timestamp("2024-06-01 01:30:00", tz="Asia/Kolkata")

    def test_request_near_midnight_rolls_into_the_next_day(self):
        # Block 96 spans 23:45-24:00 (even) -> effective at block 103 in
        # global terms, i.e. block 7 of the NEXT day (96+7-96=7), which
        # spans 01:30-01:45 the next day. Verified via plain timestamp
        # arithmetic, not a special-cased day-boundary branch.
        request = pd.Timestamp("2024-06-01 23:50:00", tz="Asia/Kolkata")
        assert block_of(request) == 96
        effective = grid_code.revision_effective_timestamp(request)
        assert effective == pd.Timestamp("2024-06-02 01:30:00", tz="Asia/Kolkata")

    def test_lead_time_bounds_hold_regardless_of_position_within_the_block(self):
        """Effective time is anchored to the REQUEST'S BLOCK START, not
        the exact request instant -- so a request landing near the end
        of an odd block has a shorter actual lead time (as little as just
        over 75 minutes: 6 blocks minus the ~15 minutes already elapsed
        in the current block) than one landing right at the block's
        start (up to 90 minutes). Both ends of that range are checked
        here, directly from the regulation's own "block, not instant"
        anchoring -- never less than 75 minutes, never more than 90."""
        block_start = pd.Timestamp("2024-06-01 13:30:00", tz="Asia/Kolkata")  # odd block (55)
        for minute_offset in [0, 5, 10, 14]:
            request = block_start + pd.Timedelta(minutes=minute_offset)
            effective = grid_code.revision_effective_timestamp(request)
            lead_time = effective - request
            assert pd.Timedelta(minutes=75) < lead_time <= pd.Timedelta(minutes=90)


class TestRevisionEligibility:
    """Regulation 49(8): WS seller revision only for bilateral transactions."""

    def test_bilateral_can_revise(self):
        assert grid_code.can_revise_schedule("bilateral") is True

    def test_collective_cannot_revise(self):
        assert grid_code.can_revise_schedule("collective") is False

    def test_unknown_transaction_type_cannot_revise(self):
        assert grid_code.can_revise_schedule("something_else") is False


class TestRealTimeMarketGateClosure:
    """Regulation 49(1)(q): RTM trades in half-hour delivery windows; the
    bid window opens 75 minutes and closes (gate closure) 60 minutes
    before each window starts. Verified against the regulation's own
    concrete instance: the 00:00-00:30 delivery window's bid window is
    22:45-23:00 hrs of the previous day."""

    def test_matches_the_regulations_own_concrete_instance(self):
        # Any timestamp within the 00:00-00:30 window should resolve the
        # same way -- using 00:10 here, not 00:00, to prove this isn't
        # just echoing the window boundary back.
        delivery = pd.Timestamp("2024-06-02 00:10:00", tz="Asia/Kolkata")
        assert grid_code.rtm_delivery_window_start(delivery) == pd.Timestamp(
            "2024-06-02 00:00:00", tz="Asia/Kolkata"
        )
        assert grid_code.rtm_bid_window_open_timestamp(delivery) == pd.Timestamp(
            "2024-06-01 22:45:00", tz="Asia/Kolkata"
        )
        assert grid_code.rtm_gate_closure_timestamp(delivery) == pd.Timestamp(
            "2024-06-01 23:00:00", tz="Asia/Kolkata"
        )

    def test_windows_repeat_every_half_hour_as_the_regulation_states(self):
        # "will be repeated every half an hour thereafter" -- check a
        # window well into the day, not just the first one.
        delivery = pd.Timestamp("2024-06-01 14:22:00", tz="Asia/Kolkata")
        assert grid_code.rtm_delivery_window_start(delivery) == pd.Timestamp(
            "2024-06-01 14:00:00", tz="Asia/Kolkata"
        )
        assert grid_code.rtm_gate_closure_timestamp(delivery) == pd.Timestamp(
            "2024-06-01 13:00:00", tz="Asia/Kolkata"
        )

    def test_second_half_hour_window_offsets_correctly(self):
        # 14:35 falls in the 14:30-15:00 window, not 14:00-14:30.
        delivery = pd.Timestamp("2024-06-01 14:35:00", tz="Asia/Kolkata")
        assert grid_code.rtm_delivery_window_start(delivery) == pd.Timestamp(
            "2024-06-01 14:30:00", tz="Asia/Kolkata"
        )
        assert grid_code.rtm_gate_closure_timestamp(delivery) == pd.Timestamp(
            "2024-06-01 13:30:00", tz="Asia/Kolkata"
        )

    def test_bid_window_is_15_minutes_wide(self):
        delivery = pd.Timestamp("2024-06-01 09:05:00", tz="Asia/Kolkata")
        open_ts = grid_code.rtm_bid_window_open_timestamp(delivery)
        gate_ts = grid_code.rtm_gate_closure_timestamp(delivery)
        assert gate_ts - open_ts == pd.Timedelta(minutes=15)
