"""
test_quality_gate.py
----------------------
Roadmap P2.2 acceptance criterion: "The customer sees a quality report
before they see a forecast." Each test engineers a synthetic dataframe
with EXACTLY ONE injected problem (checked to not accidentally trip a
different check too), so a check firing (or not firing) is unambiguous.

RUN WITH:
  pytest tests/test_quality_gate.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.quality.gate import run_quality_checks
from src.utils.config_loader import get_plant_config

PLANT = get_plant_config("jaipur_100mw")
CAPACITY_MW = PLANT["capacity"]["ac_capacity_mw"]


def _clean_day(date: str = "2024-06-01") -> pd.DataFrame:
    """A full clean day, realistic solar curve, zero at night -- the
    baseline every dirty-data test starts from and mutates ONE thing in."""
    idx = pd.date_range(date, periods=96, freq="15min", tz="Asia/Kolkata")
    hours = idx.hour + idx.minute / 60
    solar_factor = np.clip(np.cos((hours - 12.5) * np.pi / 12), 0, None)
    power = 0.7 * CAPACITY_MW * solar_factor
    return pd.DataFrame({"solar_output_mw": power}, index=idx)


class TestCleanDataPasses:
    def test_clean_day_has_no_issues(self):
        report = run_quality_checks(_clean_day(), PLANT)
        assert report.passed is True
        assert report.issues == []


class TestMissingTimezone:
    def test_naive_index_is_an_error(self):
        df = _clean_day()
        df.index = df.index.tz_localize(None)
        report = run_quality_checks(df, PLANT)
        assert report.passed is False
        checks = {i.check for i in report.issues}
        assert "missing_timezone" in checks


class TestDuplicateTimestamps:
    def test_duplicated_row_is_flagged(self):
        df = _clean_day()
        # Duplicate a midday (non-boundary) row so it can't be confused
        # with any other check.
        dup = df.iloc[[48]]
        df = pd.concat([df, dup]).sort_index()
        report = run_quality_checks(df, PLANT)
        issue = next(i for i in report.issues if i.check == "duplicate_timestamps")
        assert issue.severity == "error"
        assert issue.count == 2  # both copies of the duplicated timestamp


class TestTimestampGaps:
    def test_dropped_row_is_flagged_as_a_gap(self):
        df = _clean_day()
        df = df.drop(df.index[70])  # midday, unambiguous
        report = run_quality_checks(df, PLANT)
        issue = next(i for i in report.issues if i.check == "timestamp_gaps")
        assert issue.severity == "warning"
        assert issue.count == 1


class TestNegativePower:
    def test_negative_reading_is_an_error(self):
        df = _clean_day()
        df.iloc[40, 0] = -3.0  # midday -- would otherwise be well above zero
        report = run_quality_checks(df, PLANT)
        issue = next(i for i in report.issues if i.check == "negative_power")
        assert issue.severity == "error"
        assert issue.count == 1

    def test_tiny_negative_noise_is_not_flagged(self):
        df = _clean_day()
        df.iloc[0, 0] = -0.001  # floating-point noise around a true zero at night
        report = run_quality_checks(df, PLANT)
        assert not any(i.check == "negative_power" for i in report.issues)


class TestAboveCapacity:
    def test_reading_above_capacity_is_an_error(self):
        df = _clean_day()
        df.iloc[48, 0] = CAPACITY_MW * 1.5  # midday, unambiguous
        report = run_quality_checks(df, PLANT)
        issue = next(i for i in report.issues if i.check == "above_capacity")
        assert issue.severity == "error"
        assert issue.count == 1

    def test_reading_within_tolerance_is_not_flagged(self):
        df = _clean_day()
        df.iloc[48, 0] = CAPACITY_MW * 1.01  # inside the 2% tolerance
        report = run_quality_checks(df, PLANT)
        assert not any(i.check == "above_capacity" for i in report.issues)


class TestFlatline:
    def test_four_identical_consecutive_readings_is_flagged(self):
        df = _clean_day()
        df.iloc[48:52, 0] = 42.0  # midday, 4 identical blocks = 1 hour
        report = run_quality_checks(df, PLANT)
        issue = next(i for i in report.issues if i.check == "flatline")
        assert issue.severity == "warning"
        assert issue.count == 4

    def test_three_identical_readings_is_below_the_threshold(self):
        df = _clean_day()
        df.iloc[48:51, 0] = 42.0  # only 3 -- below DEFAULT_MIN_FLATLINE_BLOCKS
        report = run_quality_checks(df, PLANT)
        assert not any(i.check == "flatline" for i in report.issues)

    def test_repeated_zeros_at_night_are_not_a_flatline(self):
        """Every clean night already has ~40 identical zero readings in a
        row -- that must never be flagged; it's the expected, normal case."""
        report = run_quality_checks(_clean_day(), PLANT)
        assert not any(i.check == "flatline" for i in report.issues)


class TestNightTimeNonZero:
    def test_generation_reported_at_midnight_is_an_error(self):
        df = _clean_day()
        df.iloc[2, 0] = 10.0  # 00:30 -- unambiguously night at this latitude
        report = run_quality_checks(df, PLANT)
        issue = next(i for i in report.issues if i.check == "night_time_non_zero")
        assert issue.severity == "error"
        assert issue.count == 1

    def test_below_threshold_noise_at_night_is_not_flagged(self):
        df = _clean_day()
        df.iloc[2, 0] = 0.1  # below DEFAULT_NIGHT_TIME_THRESHOLD_MW
        report = run_quality_checks(df, PLANT)
        assert not any(i.check == "night_time_non_zero" for i in report.issues)


class TestReportShape:
    def test_summary_and_to_dict_do_not_crash_on_a_dirty_report(self):
        df = _clean_day()
        df.iloc[40, 0] = -3.0
        report = run_quality_checks(df, PLANT)
        assert isinstance(report.summary(), str)
        d = report.to_dict()
        assert d["passed"] is False
        assert d["plant_id"] == "jaipur_100mw"
        assert len(d["issues"]) >= 1
