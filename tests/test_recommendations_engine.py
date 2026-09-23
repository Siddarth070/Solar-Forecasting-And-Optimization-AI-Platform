"""
test_recommendations_engine.py
---------------------------------
Roadmap P2.6 acceptance criteria: human-approved-only recommendations,
each with a trigger and evidence, derived purely from already-computed
P2.4/P2.5 reports -- never automatic control, and never a recommendation
without the numbers that produced it.

RUN WITH:
  pytest tests/test_recommendations_engine.py -v
"""

import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.attribution.loss import attribute_losses, expected_generation_mw
from src.features.pipeline import clear_sky_ghi
from src.recommendations.engine import (
    TYPE_BATTERY_ACTION,
    TYPE_INSPECTION,
    TYPE_SCHEDULE_REVISION,
    recommend_battery_actions,
    recommend_inspections,
    recommend_schedule_revisions,
)
from src.risk.schedule_risk import score_schedule_risk
from src.utils.config_loader import get_plant_config

PLANT = get_plant_config("jaipur_100mw")
PRE_CUTOVER = dt.date(2024, 6, 1)
DT_HOURS = 0.25


def _clear_day_loss_input():
    idx = pd.date_range("2024-06-01", periods=96, freq="15min", tz="Asia/Kolkata")
    location = PLANT["location"]
    ghi = clear_sky_ghi(idx, location["latitude"], location["longitude"], location.get("elevation_m", 0.0))
    df = pd.DataFrame({"shortwave_radiation": ghi.to_numpy(), "temperature_2m": 25.0}, index=idx)
    df["solar_output_mw"] = expected_generation_mw(df, PLANT)
    return df


class TestScheduleRevisionRecommendations:
    def _risk_report(self, timestamps=None):
        return score_schedule_risk(
            declared_schedule_mw=[50, 50],
            p10_mw=[48, 48], p50_mw=[50, 50], p90_mw=[70, 70],  # both blocks high risk (band 3 tail)
            dt_hours=DT_HOURS, plant_config=PLANT, as_of=PRE_CUTOVER,
            timestamps=timestamps,
        )

    def test_high_risk_block_without_timestamps_is_recommended_with_unchecked_gate_note(self):
        report = self._risk_report()
        recs = recommend_schedule_revisions(report, PLANT, now=None)
        assert len(recs) == 2
        assert all(r["recommendation_type"] == TYPE_SCHEDULE_REVISION for r in recs)
        assert "not checked" in recs[0]["evidence"]["gate_note"]

    def test_block_before_the_effective_timestamp_is_locked_and_skipped(self):
        # now=13:56 -> effective_timestamp is 15:30 (CERC 14/SM/2023's own
        # worked example, verified in test_grid_code.py). A block at 15:15
        # is locked (can't be revised); one at 15:30 is still open.
        timestamps = [pd.Timestamp("2024-06-01T15:15:00+05:30"), pd.Timestamp("2024-06-01T15:30:00+05:30")]
        report = self._risk_report(timestamps=timestamps)
        recs = recommend_schedule_revisions(report, PLANT, now="2024-06-01T13:56:00+05:30")
        assert len(recs) == 1
        assert recs[0]["timestamp"] == "2024-06-01T15:30:00+05:30"
        assert "still revisable" in recs[0]["evidence"]["gate_note"]

    def test_collective_transaction_type_is_never_recommended(self):
        collective_plant = {
            **PLANT,
            "regulatory": {**PLANT["regulatory"], "revision_windows": {"transaction_type": "collective"}},
        }
        report = self._risk_report()
        assert recommend_schedule_revisions(report, collective_plant, now=None) == []

    def test_low_risk_blocks_are_never_recommended(self):
        report = score_schedule_risk(
            declared_schedule_mw=[50], p10_mw=[49], p50_mw=[50], p90_mw=[51],
            dt_hours=DT_HOURS, plant_config=PLANT, as_of=PRE_CUTOVER,
        )
        assert recommend_schedule_revisions(report, PLANT, now=None) == []


class TestBatteryActionRecommendations:
    FULL_BATTERY = {"soc_mwh": 10.0, "capacity_mwh": 20.0, "charge_rate_mw": 25.0, "discharge_rate_mw": 25.0}

    def test_no_battery_state_yields_no_recommendations(self):
        report = score_schedule_risk(
            declared_schedule_mw=[50], p10_mw=[36], p50_mw=[38], p90_mw=[50],
            dt_hours=DT_HOURS, plant_config=PLANT, as_of=PRE_CUTOVER,
        )
        assert recommend_battery_actions(report, None, DT_HOURS) == []

    def test_under_injection_recommends_discharge_with_exact_mwh(self):
        # p50 deviation = (38-50)*0.25 = -3.0 MWh (under-injection).
        report = score_schedule_risk(
            declared_schedule_mw=[50], p10_mw=[36], p50_mw=[38], p90_mw=[50],
            dt_hours=DT_HOURS, plant_config=PLANT, as_of=PRE_CUTOVER,
        )
        recs = recommend_battery_actions(report, self.FULL_BATTERY, DT_HOURS)
        assert len(recs) == 1
        rec = recs[0]
        assert rec["recommendation_type"] == TYPE_BATTERY_ACTION
        assert rec["evidence"]["direction"] == "discharge"
        assert rec["evidence"]["p50_deviation_mwh"] == pytest.approx(-3.0)
        # headroom = min(soc=10, discharge_rate*dt=25*0.25=6.25) = 6.25 >= 3.0 needed
        assert rec["evidence"]["achievable_mwh"] == pytest.approx(3.0)
        assert rec["evidence"]["partial"] is False

    def test_insufficient_headroom_is_reported_as_partial(self):
        report = score_schedule_risk(
            declared_schedule_mw=[50], p10_mw=[36], p50_mw=[38], p90_mw=[50],
            dt_hours=DT_HOURS, plant_config=PLANT, as_of=PRE_CUTOVER,
        )
        low_battery = {**self.FULL_BATTERY, "soc_mwh": 1.0}
        recs = recommend_battery_actions(report, low_battery, DT_HOURS)
        assert len(recs) == 1
        assert recs[0]["evidence"]["achievable_mwh"] == pytest.approx(1.0)
        assert recs[0]["evidence"]["partial"] is True
        assert "insufficient" in recs[0]["suggested_action"]

    def test_over_injection_recommends_charge(self):
        # p90 deviation = (70-50)*0.25 = +5.0 MWh (over-injection) -- use
        # the earlier high-risk scenario, whose p50 deviation is 0, so
        # force a nonzero p50 to actually trigger a recommendation.
        report = score_schedule_risk(
            declared_schedule_mw=[50], p10_mw=[50], p50_mw=[58], p90_mw=[70],
            dt_hours=DT_HOURS, plant_config=PLANT, as_of=PRE_CUTOVER,
        )
        recs = recommend_battery_actions(report, self.FULL_BATTERY, DT_HOURS)
        assert len(recs) == 1
        assert recs[0]["evidence"]["direction"] == "charge"
        assert recs[0]["evidence"]["p50_deviation_mwh"] == pytest.approx(2.0)  # (58-50)*0.25

    def test_low_risk_block_is_never_recommended_even_with_full_battery(self):
        report = score_schedule_risk(
            declared_schedule_mw=[50], p10_mw=[49], p50_mw=[50], p90_mw=[51],
            dt_hours=DT_HOURS, plant_config=PLANT, as_of=PRE_CUTOVER,
        )
        assert recommend_battery_actions(report, self.FULL_BATTERY, DT_HOURS) == []


class TestInspectionRecommendations:
    def test_one_equipment_run_yields_one_recommendation(self):
        df = _clear_day_loss_input()
        col = df.columns.get_loc("solar_output_mw")
        for i in range(40, 44):
            df.iloc[i, col] -= 15.0
        report = attribute_losses(df, PLANT)
        recs = recommend_inspections(report)
        assert len(recs) == 1
        rec = recs[0]
        assert rec["recommendation_type"] == TYPE_INSPECTION
        assert rec["evidence"]["run_length_blocks"] == 4
        assert rec["evidence"]["total_loss_mw_summed_across_run"] == pytest.approx(60.0)
        assert len(rec["evidence"]["block_timestamps"]) == 4

    def test_two_separate_equipment_runs_yield_two_recommendations(self):
        df = _clear_day_loss_input()
        col = df.columns.get_loc("solar_output_mw")
        for i in range(40, 44):
            df.iloc[i, col] -= 15.0
        for i in range(64, 68):
            df.iloc[i, col] -= 12.0
        report = attribute_losses(df, PLANT)
        recs = recommend_inspections(report)
        assert len(recs) == 2
        starts = {r["evidence"]["run_start_timestamp"] for r in recs}
        assert len(starts) == 2

    def test_curtailment_blocks_do_not_trigger_an_inspection(self):
        df = _clear_day_loss_input()
        col = df.columns.get_loc("solar_output_mw")
        df.iloc[40:44, col] = 40.0  # flat clip -> curtailment, not equipment
        report = attribute_losses(df, PLANT)
        assert recommend_inspections(report) == []

    def test_no_material_loss_yields_no_recommendations(self):
        report = attribute_losses(_clear_day_loss_input(), PLANT)
        assert recommend_inspections(report) == []
