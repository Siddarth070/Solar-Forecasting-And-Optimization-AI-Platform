"""
test_multi_plant.py
--------------------
Roadmap P1.1 acceptance criterion, tested directly: "No module reads a
global location. Two plants in two different states run side by side from
the same binary."

RUN WITH:
  pytest tests/test_multi_plant.py -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.pipeline import build_features, SERVING_FEATURE_COLUMNS
from src.utils.config_loader import get_plant_config, list_plant_ids

PLANT_A, PLANT_B = "jaipur_100mw", "pune_50mw"


def _weather_frame():
    index = pd.date_range("2024-06-01", periods=24, freq="h", tz="Asia/Kolkata")
    return pd.DataFrame(
        {
            "hour": index.hour,
            "month": index.month,
            "shortwave_radiation": 500.0,
            "cloud_cover": 20.0,
            "temperature_2m": 32.0,
            "relative_humidity_2m": 35.0,
            "wind_speed_10m": 3.0,
        },
        index=index,
    )


class TestMultiPlant:
    def test_at_least_two_plants_in_two_different_states(self):
        assert {PLANT_A, PLANT_B}.issubset(set(list_plant_ids()))
        a, b = get_plant_config(PLANT_A), get_plant_config(PLANT_B)
        assert a["location"]["state"] != b["location"]["state"]

    def test_plants_have_different_locations_and_capacities(self):
        a, b = get_plant_config(PLANT_A), get_plant_config(PLANT_B)
        assert a["location"]["latitude"] != b["location"]["latitude"]
        assert a["location"]["longitude"] != b["location"]["longitude"]
        assert a["capacity"]["ac_capacity_mw"] != b["capacity"]["ac_capacity_mw"]

    def test_same_process_serves_both_plants_with_different_clear_sky(self):
        """The real proof: run both plants' features through build_features
        IN THE SAME PYTHON PROCESS, for the identical weather input and
        identical timestamps, and confirm the plant-specific clear-sky
        calculation actually differs -- not just that the config dicts
        differ, but that the computation genuinely uses each one."""
        weather = _weather_frame()
        plant_a, plant_b = get_plant_config(PLANT_A), get_plant_config(PLANT_B)

        features_a = build_features(weather, plant_a)
        features_b = build_features(weather, plant_b)

        # Same timestamps, same input weather, different location -> the
        # solar-position clear-sky model must produce different values.
        assert not features_a["clear_sky_ghi_model"].equals(features_b["clear_sky_ghi_model"])

    def test_no_serving_column_leaks_between_plants(self):
        """Calling build_features for plant B right after plant A must not
        leave any stale state affecting plant A's own columns (guards
        against a module-level global location cache bleeding across
        requests, which is exactly what P1.1 forbids)."""
        weather = _weather_frame()
        plant_a, plant_b = get_plant_config(PLANT_A), get_plant_config(PLANT_B)

        first_a = build_features(weather, plant_a)[SERVING_FEATURE_COLUMNS].copy()
        build_features(weather, plant_b)  # interleave a different plant
        second_a = build_features(weather, plant_a)[SERVING_FEATURE_COLUMNS]

        pd.testing.assert_frame_equal(first_a, second_a)
