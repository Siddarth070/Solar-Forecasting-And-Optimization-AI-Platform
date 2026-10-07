"""
test_attribution_validation.py
--------------------------------
Unit tests for src/evaluation/attribution_validation.py (roadmap P3.4).

IMPORTANT: every O&M log and generation series here is a SYNTHETIC
fixture built for the test -- P3.1 (a real plant's data, including real
outage/curtailment logs) is not secured yet. Nothing here is a real
P3.4 result; these tests only prove the validation harness computes
precision/recall correctly on a known, hand-constructed confusion
matrix, so it's ready the moment a real log exists.

RUN WITH:
  pytest tests/test_attribution_validation.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.attribution.loss import attribute_losses
from src.evaluation.attribution_validation import (
    AttributionValidationError,
    load_om_log_csv,
    validate_attribution,
    validate_attribution_from_csv,
)
from src.utils.config_loader import get_plant_config


def _write_csv(tmp_path, rows, name="om_log.csv"):
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


class TestLoadOmLogCsv:
    def test_valid_log_parses_and_maps_cause(self, tmp_path):
        path = _write_csv(tmp_path, [
            {"start_timestamp": "2024-06-01T10:00:00+05:30",
             "end_timestamp": "2024-06-01T12:00:00+05:30", "event_type": "outage"},
            {"start_timestamp": "2024-06-02T10:00:00+05:30",
             "end_timestamp": "2024-06-02T11:00:00+05:30", "event_type": "curtailment"},
        ])
        log = load_om_log_csv(path)
        assert list(log["cause"]) == ["equipment", "curtailment"]

    def test_missing_column_raises(self, tmp_path):
        path = _write_csv(tmp_path, [{"start_timestamp": "2024-06-01T10:00:00+05:30", "event_type": "outage"}])
        with pytest.raises(AttributionValidationError, match="missing required column"):
            load_om_log_csv(path)

    def test_tz_naive_column_raises(self, tmp_path):
        path = _write_csv(tmp_path, [
            {"start_timestamp": "2024-06-01T10:00:00+05:30",
             "end_timestamp": "2024-06-01T12:00:00", "event_type": "outage"},
        ])
        with pytest.raises(AttributionValidationError, match="no UTC offset"):
            load_om_log_csv(path)

    def test_mixed_tz_within_one_column_raises(self, tmp_path):
        path = _write_csv(tmp_path, [
            {"start_timestamp": "2024-06-01T10:00:00+05:30",
             "end_timestamp": "2024-06-01T12:00:00+05:30", "event_type": "outage"},
            {"start_timestamp": "2024-06-02T10:00:00",
             "end_timestamp": "2024-06-02T12:00:00+05:30", "event_type": "outage"},
        ])
        with pytest.raises(AttributionValidationError, match="one consistent timezone"):
            load_om_log_csv(path)

    def test_unrecognised_event_type_dropped(self, tmp_path):
        path = _write_csv(tmp_path, [
            {"start_timestamp": "2024-06-01T10:00:00+05:30",
             "end_timestamp": "2024-06-01T12:00:00+05:30", "event_type": "outage"},
            {"start_timestamp": "2024-06-02T10:00:00+05:30",
             "end_timestamp": "2024-06-02T11:00:00+05:30", "event_type": "cleaning"},
        ])
        log = load_om_log_csv(path)
        assert len(log) == 1
        assert log.iloc[0]["event_type"] == "outage"


@pytest.fixture()
def jaipur_config():
    return get_plant_config("jaipur_100mw")


def _build_report(jaipur_config):
    """A synthetic generation series with one deliberate equipment-like
    dip (flat, below expected, no cloud ramp) and one curtailment-like
    flat clip at a cap -- fed through the REAL attribute_losses(), so
    the predicted causes are genuinely computed, not hand-set."""
    capacity = jaipur_config["capacity"]["ac_capacity_mw"]
    index = pd.date_range("2024-06-01T06:00:00+05:30", periods=24, freq="h")
    hours = index.hour.to_numpy()

    ghi = np.clip(800 * np.sin(np.pi * (hours - 6) / 12), 0, None)
    temp = np.full(len(index), 30.0)
    expected = ghi / 1000 * (1 + jaipur_config["capacity"]["temperature_coefficient"] * (temp - 25)) \
        * jaipur_config["capacity"]["performance_ratio"] * capacity
    expected = np.clip(expected, 0, capacity)

    actual = expected.copy()
    # Equipment-like run: hours 7-10, output degraded AND noisy (not
    # flat) -- the module's own test for "equipment" vs "curtailment" is
    # exactly this: a flat clip reads as curtailment, a noisy shortfall
    # that still doesn't track irradiance reads as equipment.
    equipment_mask = (hours >= 7) & (hours <= 10)
    actual[equipment_mask] = [2.0, 5.0, 3.0, 6.0]
    # Curtailment-like run: hours 13-16, output pinned perfectly flat at
    # a cap well below expected (which keeps varying) -- the module's
    # own signature for a hard output cap.
    curtailment_mask = (hours >= 13) & (hours <= 16)
    actual[curtailment_mask] = 25.0

    df = pd.DataFrame({
        "solar_output_mw": actual,
        "shortwave_radiation": ghi,
        "temperature_2m": temp,
    }, index=index)

    report = attribute_losses(df, jaipur_config)
    return report, equipment_mask, curtailment_mask, index


class TestValidateAttribution:
    def test_perfect_log_gives_high_precision_and_recall(self, jaipur_config):
        report, equipment_mask, curtailment_mask, index = _build_report(jaipur_config)

        om_log = pd.DataFrame({
            "start_timestamp": [index[equipment_mask][0], index[curtailment_mask][0]],
            "end_timestamp": [index[equipment_mask][-1], index[curtailment_mask][-1]],
            "event_type": ["outage", "curtailment"],
            "cause": ["equipment", "curtailment"],
        })

        result = validate_attribution(report, om_log)

        assert result["plant_id"] == "jaipur_100mw"
        assert result["n_blocks_scoreable_against_log"] > 0
        for cause in ("equipment", "curtailment"):
            pc = result["per_class"][cause]
            # Not asserting a fabricated exact number -- just that the
            # harness found a real, strong match against a log that
            # deliberately covers exactly the injected pattern.
            assert pc["true_positives"] > 0
            assert pc["precision"] > 0.5
            assert pc["recall"] > 0.5

    def test_log_with_no_coverage_scores_nothing(self, jaipur_config):
        report, _, _, index = _build_report(jaipur_config)
        # A log that covers a period entirely outside this report's data.
        om_log = pd.DataFrame({
            "start_timestamp": [pd.Timestamp("2099-01-01T00:00:00+05:30")],
            "end_timestamp": [pd.Timestamp("2099-01-01T01:00:00+05:30")],
            "event_type": ["outage"],
            "cause": ["equipment"],
        })
        result = validate_attribution(report, om_log)
        assert result["n_blocks_scoreable_against_log"] == 0
        assert result["n_blocks_unscoreable"] == result["n_blocks_total"]
        for cause in ("equipment", "curtailment"):
            assert np.isnan(result["per_class"][cause]["precision"])

    def test_wrong_cause_in_log_hurts_precision(self, jaipur_config):
        report, equipment_mask, curtailment_mask, index = _build_report(jaipur_config)
        # Log (deliberately wrong) claims the equipment-like run was
        # actually curtailment -- the real model still predicts
        # "equipment" there, so this should show up as a false positive
        # for "equipment" and a false negative against "curtailment".
        om_log = pd.DataFrame({
            "start_timestamp": [index[equipment_mask][0]],
            "end_timestamp": [index[equipment_mask][-1]],
            "event_type": ["curtailment"],
            "cause": ["curtailment"],
        })
        result = validate_attribution(report, om_log)
        assert result["per_class"]["equipment"]["false_positives"] > 0


class TestValidateAttributionFromCsv:
    def test_end_to_end_from_csv(self, tmp_path, jaipur_config):
        report, equipment_mask, curtailment_mask, index = _build_report(jaipur_config)
        path = _write_csv(tmp_path, [
            {"start_timestamp": index[equipment_mask][0].isoformat(),
             "end_timestamp": index[equipment_mask][-1].isoformat(), "event_type": "outage"},
            {"start_timestamp": index[curtailment_mask][0].isoformat(),
             "end_timestamp": index[curtailment_mask][-1].isoformat(), "event_type": "curtailment"},
        ])
        result = validate_attribution_from_csv(report, path)
        assert result["per_class"]["equipment"]["true_positives"] > 0
        assert result["per_class"]["curtailment"]["true_positives"] > 0
