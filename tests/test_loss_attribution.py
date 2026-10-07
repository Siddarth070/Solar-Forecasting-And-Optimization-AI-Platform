"""
test_loss_attribution.py
--------------------------
Roadmap P2.4 acceptance criterion: "Rule-based with explicit thresholds
and a confidence label. Never outputs a cause without the evidence that
produced it." Each dirty-data test engineers exactly one kind of loss
(a single-block cloud transient, a multi-block hardware-style step, a
multi-block hard flat clip, an isolated unexplained dip, or a slow
multi-day drift) so a classification landing on one specific cause is
unambiguous, with hand-computed expected numbers, not just "it ran".

RUN WITH:
  pytest tests/test_loss_attribution.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.attribution.loss import (
    CAUSE_CURTAILMENT,
    CAUSE_EQUIPMENT,
    CAUSE_UNKNOWN,
    CAUSE_WEATHER,
    attribute_losses,
    daily_performance_ratio,
    detect_soiling,
    expected_generation_mw,
)
from src.features.pipeline import clear_sky_ghi
from src.utils.config_loader import get_plant_config

PLANT = get_plant_config("jaipur_100mw")
CAPACITY_MW = PLANT["capacity"]["ac_capacity_mw"]


def _clear_day(date: str = "2024-06-01") -> pd.DataFrame:
    """A full, perfectly clear day: measured GHI equals the clear-sky
    model exactly (clear_sky_index == 1.0 at every daytime block) and a
    constant 25 degC temperature (temp_factor == 1 identically) -- this
    keeps expected_generation_mw's arithmetic exact and hand-checkable,
    and actual is set equal to expected, so the baseline has zero loss
    anywhere until a test deliberately injects one."""
    idx = pd.date_range(date, periods=96, freq="15min", tz="Asia/Kolkata")
    location = PLANT["location"]
    ghi = clear_sky_ghi(idx, location["latitude"], location["longitude"],
                         location.get("elevation_m", 0.0))
    df = pd.DataFrame({
        "shortwave_radiation": ghi.to_numpy(),
        "temperature_2m": 25.0,
    }, index=idx)
    df["solar_output_mw"] = expected_generation_mw(df, PLANT)
    return df


class TestNoMaterialLossOnAClearDay:
    def test_actual_equals_expected_everywhere_yields_no_blocks(self):
        report = attribute_losses(_clear_day(), PLANT)
        assert report.blocks == []
        assert report.causes_summary() == {}


class TestMaterialityThresholds:
    def test_tiny_loss_well_below_both_thresholds_is_not_flagged(self):
        df = _clear_day()
        col = df.columns.get_loc("solar_output_mw")
        df.iloc[48, col] -= 0.1  # noon: expected ~78 MW -- 0.1 MW is far below 1 MW / 5%
        report = attribute_losses(df, PLANT)
        assert report.blocks == []

    def test_loss_at_dawn_is_skipped_despite_being_100_percent_of_expected(self):
        df = _clear_day()
        col = df.columns.get_loc("solar_output_mw")
        expected_at_dawn = df["solar_output_mw"].iloc[24]  # 06:00, ~1.18 MW
        assert 0 < expected_at_dawn < 0.05 * CAPACITY_MW  # below the classification floor
        df.iloc[24, col] = 0.0  # 100% loss fraction, yet still skipped
        report = attribute_losses(df, PLANT)
        assert report.blocks == []


class TestWeatherCause:
    def test_single_block_cloud_transient_is_weather(self):
        df = _clear_day()
        ghi_col = df.columns.get_loc("shortwave_radiation")
        power_col = df.columns.get_loc("solar_output_mw")

        # A passing cloud at noon: GHI drops to 30% of its clear-sky value
        # for exactly one block, and actual output is ALSO 5 MW below what
        # that reduced irradiance alone would predict (the extra loss a
        # real inverter's MPPT lag during a fast transient would cause).
        df.iloc[48, ghi_col] *= 0.3
        recomputed_expected_48 = expected_generation_mw(df, PLANT).iloc[48]
        df.iloc[48, power_col] = recomputed_expected_48 - 5.0

        report = attribute_losses(df, PLANT)
        assert len(report.blocks) == 1
        block = report.blocks[0]
        assert block.cause == CAUSE_WEATHER
        assert block.confidence == "medium"
        assert block.loss_mw == pytest.approx(5.0, abs=0.01)
        assert block.evidence["run_length_blocks"] == 1
        assert block.evidence["clear_sky_index"] == pytest.approx(0.3, abs=0.01)
        assert block.evidence["clear_sky_index_ramp"] >= 0.15


class TestEquipmentCause:
    def test_sustained_proportional_drop_with_stable_irradiance_is_equipment(self):
        df = _clear_day()
        col = df.columns.get_loc("solar_output_mw")
        # 4 consecutive blocks (1 hour), each exactly 15 MW below its own
        # expected value -- a constant absolute offset preserves the
        # SHAPE of the natural solar ramp (same std as `expected`), which
        # is what should read as "equipment", not "curtailment".
        for i in range(40, 44):
            df.iloc[i, col] -= 15.0

        report = attribute_losses(df, PLANT)
        assert len(report.blocks) == 4
        for block in report.blocks:
            assert block.cause == CAUSE_EQUIPMENT
            assert block.confidence == "medium"
            assert block.loss_mw == pytest.approx(15.0, abs=0.01)
            assert block.evidence["run_length_blocks"] == 4
            assert block.evidence["clear_sky_index_ramp"] < 0.15
            assert "no per-inverter" in block.evidence["reason"]


class TestCurtailmentCause:
    def test_hard_flat_clip_while_expected_still_varies_is_curtailment(self):
        df = _clear_day()
        col = df.columns.get_loc("solar_output_mw")
        expected_run = df["solar_output_mw"].iloc[40:44].copy()
        # Same 4-block window as the equipment test, but pinned to one flat
        # value instead of shifted by a constant -- expected keeps varying
        # (it's on the morning ramp) while actual does not.
        df.iloc[40:44, col] = 40.0

        report = attribute_losses(df, PLANT)
        assert len(report.blocks) == 4
        for i, block in enumerate(report.blocks):
            assert block.cause == CAUSE_CURTAILMENT
            assert block.confidence == "medium"
            assert block.loss_mw == pytest.approx(expected_run.iloc[i] - 40.0, abs=0.01)
            assert block.evidence["run_length_blocks"] == 4
            assert block.evidence["run_actual_std_mw"] == pytest.approx(0.0, abs=1e-6)
            assert block.evidence["run_expected_std_mw"] > 1.0
            assert "no real grid curtailment-instruction feed" in block.evidence["reason"]


class TestUnknownCause:
    def test_isolated_unexplained_dip_is_unknown(self):
        df = _clear_day()
        col = df.columns.get_loc("solar_output_mw")
        df.iloc[60, col] -= 10.0  # 15:00, no GHI change, no run -- just one bad block
        report = attribute_losses(df, PLANT)
        assert len(report.blocks) == 1
        block = report.blocks[0]
        assert block.cause == CAUSE_UNKNOWN
        assert block.confidence == "low"
        assert block.loss_mw == pytest.approx(10.0, abs=0.01)
        assert block.evidence["run_length_blocks"] == 1


class TestCausesSummary:
    def test_summary_aggregates_count_and_loss_by_cause(self):
        df = _clear_day()
        col = df.columns.get_loc("solar_output_mw")
        df.iloc[60, col] -= 10.0  # one "unknown" block
        report = attribute_losses(df, PLANT)
        summary = report.causes_summary()
        assert summary == {CAUSE_UNKNOWN: {"count": 1, "loss_mw": pytest.approx(10.0, abs=0.01)}}
        d = report.to_dict()
        assert d["material_loss_blocks"] == 1
        assert d["plant_id"] == "jaipur_100mw"


class TestDailyPerformanceRatioAndSoiling:
    def _three_day_df(self, day_ratios: list[float]) -> pd.DataFrame:
        idx = pd.date_range("2024-06-01", periods=96 * len(day_ratios), freq="15min", tz="Asia/Kolkata")
        location = PLANT["location"]
        ghi = clear_sky_ghi(idx, location["latitude"], location["longitude"],
                             location.get("elevation_m", 0.0))
        df = pd.DataFrame({"shortwave_radiation": ghi.to_numpy(), "temperature_2m": 25.0}, index=idx)
        expected = expected_generation_mw(df, PLANT)
        day_index = idx.normalize()
        unique_days = day_index.unique()
        scale = pd.Series(0.0, index=idx)
        for day, ratio in zip(unique_days, day_ratios):
            scale[day_index == day] = ratio
        df["solar_output_mw"] = expected * scale
        return df

    def test_daily_ratio_matches_the_injected_per_day_scale(self):
        df = self._three_day_df([1.0, 0.9, 0.8])
        daily = daily_performance_ratio(df, PLANT)
        assert len(daily) == 3
        np.testing.assert_allclose(daily.to_numpy(), [1.0, 0.9, 0.8], atol=1e-6)

    def test_sustained_decline_over_enough_days_is_soiling(self):
        daily_ratio = pd.Series(
            np.linspace(1.0, 0.85, 14),
            index=pd.date_range("2024-06-01", periods=14, freq="D"),
        )
        finding = detect_soiling(daily_ratio)
        assert finding is not None
        assert finding.confidence == "medium"
        assert finding.slope_pct_per_day < -0.1
        assert finding.days_observed == 14
        assert finding.evidence["ratio_start"] == pytest.approx(1.0, abs=1e-6)
        assert finding.evidence["ratio_end"] == pytest.approx(0.85, abs=1e-6)

    def test_flat_ratio_is_not_soiling(self):
        daily_ratio = pd.Series(
            np.full(14, 0.9),
            index=pd.date_range("2024-06-01", periods=14, freq="D"),
        )
        assert detect_soiling(daily_ratio) is None

    def test_too_few_days_returns_none_even_with_a_steep_decline(self):
        daily_ratio = pd.Series(
            np.linspace(1.0, 0.5, 5),
            index=pd.date_range("2024-06-01", periods=5, freq="D"),
        )
        assert detect_soiling(daily_ratio) is None
