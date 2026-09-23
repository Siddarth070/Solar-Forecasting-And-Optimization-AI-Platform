"""
test_schedule_risk.py
------------------------
Roadmap P2.5 acceptance criteria: a per-block risk score derived from
comparing P10/P50/P90 against the declared schedule and the real CERC
DSM volume-limit bands (src/regulatory/dsm.py), and every output must
carry the roadmap's exact disclaimer. Each scenario below hand-computes
the expected deviation MWh and band before checking the module's output
against it.

RUN WITH:
  pytest tests/test_schedule_risk.py -v
"""

import datetime as dt
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.risk.schedule_risk import (
    DISCLAIMER,
    RISK_HIGH,
    RISK_LOW,
    RISK_MEDIUM,
    score_block,
    score_schedule_risk,
)
from src.utils.config_loader import get_plant_config

PLANT = get_plant_config("jaipur_100mw")
PRE_CUTOVER = dt.date(2024, 6, 1)   # solar VL1/VL2 = 10%/15% of available capacity
POST_CUTOVER = dt.date(2026, 6, 1)  # solar VL1/VL2 = 5%/10% of available capacity
DT_HOURS = 0.25
# 100 MW * 0.25h = 25 MWh available capacity per block -- so pre-cutover
# VL1=2.5 MWh, VL2=3.75 MWh; post-cutover VL1=1.25 MWh, VL2=2.5 MWh.


class TestBandsMatchHandComputedDeviations:
    def test_all_quantiles_within_band_one_is_low_risk(self):
        # Deviations: p10 -0.5 MWh, p50 0 MWh, p90 +0.5 MWh -- all well
        # inside VL1 (2.5 MWh pre-cutover).
        block = score_block(
            declared_schedule_mw=50, p10_mw=48, p50_mw=50, p90_mw=52,
            dt_hours=DT_HOURS, plant_config=PLANT, as_of=PRE_CUTOVER,
        )
        assert block.risk_level == RISK_LOW
        assert block.worst_band == 1
        assert block.quantile_bands == {"p10": 1, "p50": 0, "p90": 1}
        assert block.quantile_deviation_mwh["p10"] == pytest.approx(-0.5)
        assert block.quantile_deviation_mwh["p50"] == pytest.approx(0.0)
        assert block.quantile_deviation_mwh["p90"] == pytest.approx(0.5)
        assert block.to_dict()["disclaimer"] == DISCLAIMER

    def test_p10_and_p50_reaching_band_two_is_medium_risk(self):
        # p10 deviation -3.5 MWh, p50 -3.0 MWh -- both between VL1 (2.5)
        # and VL2 (3.75) pre-cutover; p90 stays at exactly zero deviation.
        block = score_block(
            declared_schedule_mw=50, p10_mw=36, p50_mw=38, p90_mw=50,
            dt_hours=DT_HOURS, plant_config=PLANT, as_of=PRE_CUTOVER,
        )
        assert block.risk_level == RISK_MEDIUM
        assert block.worst_band == 2
        assert block.quantile_bands == {"p10": 2, "p50": 2, "p90": 0}
        assert block.quantile_deviation_mwh["p10"] == pytest.approx(-3.5)
        assert block.quantile_deviation_mwh["p50"] == pytest.approx(-3.0)

    def test_p90_beyond_vl2_is_high_risk_even_with_a_calm_median(self):
        # p90 deviation +5.0 MWh -- beyond VL2 (3.75) pre-cutover -- while
        # p50 sits at exactly zero deviation, showing the tail alone
        # drives the risk level, not just the median.
        block = score_block(
            declared_schedule_mw=50, p10_mw=48, p50_mw=50, p90_mw=70,
            dt_hours=DT_HOURS, plant_config=PLANT, as_of=PRE_CUTOVER,
        )
        assert block.risk_level == RISK_HIGH
        assert block.worst_band == 3
        assert block.quantile_bands == {"p10": 1, "p50": 0, "p90": 3}
        assert block.evidence["band_spread"] == 3  # from band 0 (p50) to band 3 (p90)


class TestCutoverDateChangesTheBands:
    def test_same_deviation_reads_a_higher_band_after_the_dsm_cutover(self):
        # A 2.0 MWh deviation: pre-cutover VL1=2.5 -> band 1;
        # post-cutover VL1=1.25/VL2=2.5 -> band 2. Same inputs, different
        # `as_of`, deliberately different classification -- proving the
        # real Note-1 cutover date (not a hardcoded band) drives this.
        kwargs = dict(declared_schedule_mw=50, p10_mw=50, p50_mw=42, p90_mw=50,
                      dt_hours=DT_HOURS, plant_config=PLANT)
        pre = score_block(**kwargs, as_of=PRE_CUTOVER)
        post = score_block(**kwargs, as_of=POST_CUTOVER)
        assert pre.quantile_deviation_mwh["p50"] == pytest.approx(-2.0)
        assert post.quantile_deviation_mwh["p50"] == pytest.approx(-2.0)
        assert pre.quantile_bands["p50"] == 1
        assert post.quantile_bands["p50"] == 2
        assert pre.risk_level == RISK_LOW
        assert post.risk_level == RISK_MEDIUM


class TestMissingRegulatoryConfigRaises:
    def test_plant_without_contract_rate_raises_a_clear_error(self):
        stripped = {**PLANT, "regulatory": {"seller_category": "solar"}}  # no contract_rate_rs_per_kwh
        with pytest.raises(ValueError, match="regulatory"):
            score_block(50, 48, 50, 52, DT_HOURS, stripped, PRE_CUTOVER)


class TestScheduleReport:
    def test_mismatched_list_lengths_raise(self):
        with pytest.raises(ValueError):
            score_schedule_risk([50, 50], [48], [50, 50], [52, 52], DT_HOURS, PLANT, PRE_CUTOVER)

    def test_report_aggregates_risk_levels_across_blocks(self):
        report = score_schedule_risk(
            declared_schedule_mw=[50, 50, 50],
            p10_mw=[48, 36, 48],
            p50_mw=[50, 38, 50],
            p90_mw=[52, 50, 70],
            dt_hours=DT_HOURS,
            plant_config=PLANT,
            as_of=PRE_CUTOVER,
        )
        assert len(report.blocks) == 3
        assert report.risk_summary() == {RISK_LOW: 1, RISK_MEDIUM: 1, RISK_HIGH: 1}
        d = report.to_dict()
        assert d["plant_id"] == "jaipur_100mw"
        assert d["disclaimer"] == DISCLAIMER
        assert len(d["schedule"]) == 3
