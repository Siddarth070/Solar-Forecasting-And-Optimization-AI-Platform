"""
test_dsm.py
------------
Roadmap P1.7 acceptance criterion: a config-driven DSM ruleset, tested
directly against the real CERC (Deviation Settlement Mechanism and
Related Matters) Regulations, 2024 (as amended by the First and Second
Amendments -- neither touches the WS-seller charge structure this module
implements; the Third Amendment is an unfinished DRAFT and does not
apply). See src/regulatory/dsm.py's module docstring for citations.

Every expected number below is hand-computed directly from Regulation
8(4) and its Note-1 -- not derived from the same code under test.

RUN WITH:
  pytest tests/test_dsm.py -v
"""

import datetime as dt
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.regulatory import dsm

POST_CUTOVER = dt.date(2026, 6, 1)   # after 01.04.2026
PRE_CUTOVER = dt.date(2025, 6, 1)    # before 01.04.2026


class TestVolumeLimits:
    """Note-1's literal table, both periods, both relevant categories."""

    def test_solar_post_cutover_is_5_and_10_percent(self):
        assert dsm.volume_limits("solar", POST_CUTOVER) == (0.05, 0.10)

    def test_solar_pre_cutover_is_10_and_15_percent(self):
        assert dsm.volume_limits("solar", PRE_CUTOVER) == (0.10, 0.15)

    def test_hybrid_matches_solar(self):
        assert dsm.volume_limits("hybrid", POST_CUTOVER) == dsm.volume_limits("solar", POST_CUTOVER)

    def test_wind_post_cutover_is_10_and_15_percent(self):
        assert dsm.volume_limits("wind", POST_CUTOVER) == (0.10, 0.15)

    def test_wind_pre_cutover_is_15_and_20_percent(self):
        assert dsm.volume_limits("wind", PRE_CUTOVER) == (0.15, 0.20)

    def test_cutover_boundary_is_inclusive_of_01_04_2026(self):
        assert dsm.volume_limits("solar", dsm.CUTOVER_DATE) == (0.05, 0.10)
        assert dsm.volume_limits("solar", dsm.CUTOVER_DATE - dt.timedelta(days=1)) == (0.10, 0.15)

    def test_unknown_category_raises(self):
        with pytest.raises(ValueError):
            dsm.volume_limits("nuclear", POST_CUTOVER)


class TestDeviationSettlementUnderInjection:
    """Regulation 8(4), under-injection (payable) column: contract rate /
    110% / 200% across VLwS(1) / VLwS(2) / beyond. Solar, post-cutover:
    VLwS(1)=5%, VLwS(2)=10%. available_capacity=100 MWh -> VL1=5 MWh,
    VL2=10 MWh. contract_rate=2.5 Rs/kWh -> 2500 Rs/MWh."""

    CAPACITY = 100.0
    RATE = 2.5  # Rs/kWh

    def test_within_vl1_charged_at_contract_rate(self):
        result = dsm.deviation_settlement(-3.0, self.CAPACITY, self.RATE, "solar", POST_CUTOVER)
        assert result["segment_mwh"] == (3.0, 0.0, 0.0)
        assert result["net_rs"] == pytest.approx(3.0 * 2500 * 1.00)  # 7500

    def test_spanning_vl1_and_vl2_splits_correctly(self):
        # 7 MWh under-injection: 5 in VL1 @100%, 2 in VL2 @110%.
        result = dsm.deviation_settlement(-7.0, self.CAPACITY, self.RATE, "solar", POST_CUTOVER)
        assert result["segment_mwh"] == pytest.approx((5.0, 2.0, 0.0))
        expected = 5.0 * 2500 * 1.00 + 2.0 * 2500 * 1.10
        assert result["net_rs"] == pytest.approx(expected)  # 18000

    def test_beyond_vl2_hits_200_percent_tier(self):
        # 15 MWh under-injection: 5 @100%, 5 @110%, 5 @200%.
        result = dsm.deviation_settlement(-15.0, self.CAPACITY, self.RATE, "solar", POST_CUTOVER)
        assert result["segment_mwh"] == pytest.approx((5.0, 5.0, 5.0))
        expected = 5.0 * 2500 * 1.00 + 5.0 * 2500 * 1.10 + 5.0 * 2500 * 2.00
        assert result["net_rs"] == pytest.approx(expected)  # 51250

    def test_under_injection_is_positive_net_rs_ie_payable(self):
        result = dsm.deviation_settlement(-3.0, self.CAPACITY, self.RATE, "solar", POST_CUTOVER)
        assert result["net_rs"] > 0


class TestDeviationSettlementOverInjection:
    """Regulation 8(4), over-injection (receivable) column: contract rate
    / 90% / zero across VLwS(1) / VLwS(2) / beyond."""

    CAPACITY = 100.0
    RATE = 2.5

    def test_within_vl1_receives_full_contract_rate(self):
        result = dsm.deviation_settlement(3.0, self.CAPACITY, self.RATE, "solar", POST_CUTOVER)
        assert result["net_rs"] == pytest.approx(-3.0 * 2500 * 1.00)  # -7500 (receivable)

    def test_spanning_vl1_and_vl2_gets_90_percent_on_the_second_segment(self):
        result = dsm.deviation_settlement(7.0, self.CAPACITY, self.RATE, "solar", POST_CUTOVER)
        expected = -(5.0 * 2500 * 1.00 + 2.0 * 2500 * 0.90)
        assert result["net_rs"] == pytest.approx(expected)  # -17000

    def test_beyond_vl2_receives_nothing_for_the_excess(self):
        # 15 MWh over-injection: 5 @100%, 5 @90%, 5 @0% -- the marginal
        # segment beyond VLwS(2) contributes zero, per "beyond VLwS(2) @ Zero".
        result = dsm.deviation_settlement(15.0, self.CAPACITY, self.RATE, "solar", POST_CUTOVER)
        expected = -(5.0 * 2500 * 1.00 + 5.0 * 2500 * 0.90 + 5.0 * 2500 * 0.00)
        assert result["net_rs"] == pytest.approx(expected)  # -23750

    def test_over_injection_is_negative_net_rs_ie_receivable(self):
        result = dsm.deviation_settlement(3.0, self.CAPACITY, self.RATE, "solar", POST_CUTOVER)
        assert result["net_rs"] < 0


class TestDeviationSettlementEdgeCases:
    def test_zero_deviation_settles_to_zero(self):
        result = dsm.deviation_settlement(0.0, 100.0, 2.5, "solar", POST_CUTOVER)
        assert result["net_rs"] == 0.0

    def test_pre_cutover_uses_wider_bands(self):
        # Same 7 MWh deviation, but pre-cutover VL1=10 MWh -- entirely
        # within VLwS(1), unlike the post-cutover case (VL1=5 MWh) above.
        result = dsm.deviation_settlement(-7.0, 100.0, 2.5, "solar", PRE_CUTOVER)
        assert result["segment_mwh"] == pytest.approx((7.0, 0.0, 0.0))
        assert result["net_rs"] == pytest.approx(7.0 * 2500 * 1.00)


class TestDeviationPct:
    def test_matches_regulation_6_formula(self):
        # 100 x (actual - scheduled) / available_capacity
        pct = dsm.deviation_pct(actual_injection_mwh=48.0, scheduled_generation_mwh=50.0,
                                 available_capacity_mwh=100.0)
        assert pct == pytest.approx(-2.0)

    def test_zero_available_capacity_does_not_divide_by_zero(self):
        assert dsm.deviation_pct(5.0, 0.0, 0.0) == 0.0


class TestLpCostSegments:
    """The piecewise-linear breakdown handed to the optimizer must use the
    SAME numbers as deviation_settlement (single source of truth)."""

    def test_segment_widths_and_rates_match_hand_computed_settlement(self):
        segments = dsm.lp_cost_segments(100.0, 2.5, "solar", POST_CUTOVER)
        assert segments["segment_widths_mwh"] == pytest.approx((5.0, 5.0))
        assert segments["under_injection_rates_rs_per_mwh"] == pytest.approx((2500, 2750, 5000))
        assert segments["over_injection_rates_rs_per_mwh"] == pytest.approx((2500, 2250, 0))

    def test_reconstructing_settlement_from_segments_matches_deviation_settlement(self):
        """Feeding the same segment widths/rates back through by hand
        must reproduce deviation_settlement's own net_rs -- proves the two
        functions aren't secretly using different numbers."""
        segments = dsm.lp_cost_segments(100.0, 2.5, "solar", POST_CUTOVER)
        widths = segments["segment_widths_mwh"]
        rates = segments["under_injection_rates_rs_per_mwh"]

        deviation = 12.0  # under-injection, spans all 3 segments
        remaining = deviation
        manual_rs = 0.0
        for width, rate in zip(widths, rates[:2]):
            take = min(remaining, width)
            manual_rs += take * rate
            remaining -= take
        manual_rs += remaining * rates[2]

        expected = dsm.deviation_settlement(-deviation, 100.0, 2.5, "solar", POST_CUTOVER)["net_rs"]
        assert manual_rs == pytest.approx(expected)
