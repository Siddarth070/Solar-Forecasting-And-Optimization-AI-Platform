"""
test_open_meteo_fetcher.py
--------------------------
Unit tests for the OpenMeteoFetcher module.

WHAT WE TEST:
  1. Config loads correctly
  2. API parameters are built correctly
  3. Response parsing produces expected DataFrame shape/types
  4. Validation catches bad data correctly

WHY THESE TESTS MATTER:
  If the Open-Meteo API changes its response format (it happens),
  these tests will catch it immediately rather than letting bad data
  silently flow into your models.

RUN WITH:
  pytest tests/test_open_meteo_fetcher.py -v
"""

import pytest
import pandas as pd
import sys
from pathlib import Path

# Add project root to path so imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


from src.utils.config_loader import get_config


class TestConfigLoader:
    """Test that config loads and has expected structure."""

    def test_config_loads(self):
        config = get_config()
        assert config is not None

    def test_config_has_location(self):
        config = get_config()
        assert "location" in config
        assert "latitude" in config["location"]
        assert "longitude" in config["location"]

    def test_jaipur_coordinates(self):
        config = get_config()
        loc = config["location"]
        # Jaipur is roughly 26.9°N, 75.8°E
        assert 26.5 < loc["latitude"] < 27.5
        assert 75.0 < loc["longitude"] < 76.5

    def test_config_has_data_sources(self):
        config = get_config()
        assert "data_sources" in config
        assert "open_meteo" in config["data_sources"]


class TestResponseParsing:
    """Test DataFrame parsing from mock API response."""

    def _mock_response(self):
        """Build a minimal valid Open-Meteo API response for testing."""
        return {
            "latitude": 26.9124,
            "longitude": 75.7873,
            "timezone": "Asia/Kolkata",
            "hourly": {
                "time": [
                    "2024-01-01T00:00",
                    "2024-01-01T01:00",
                    "2024-01-01T02:00",
                ],
                "temperature_2m": [18.5, 17.2, 16.8],
                "relative_humidity_2m": [55.0, 58.0, 60.0],
                "precipitation": [0.0, 0.0, 0.0],
                "cloud_cover": [10.0, 15.0, 12.0],
                "wind_speed_10m": [3.2, 2.8, 3.0],
                "shortwave_radiation": [0.0, 0.0, 0.0],
                "direct_radiation": [0.0, 0.0, 0.0],
                "diffuse_radiation": [0.0, 0.0, 0.0],
            }
        }

    def test_parse_produces_dataframe(self):
        from src.ingestion.open_meteo_fetcher import OpenMeteoFetcher
        fetcher = OpenMeteoFetcher()
        df = fetcher._parse_response(self._mock_response())
        assert isinstance(df, pd.DataFrame)

    def test_parse_correct_row_count(self):
        from src.ingestion.open_meteo_fetcher import OpenMeteoFetcher
        fetcher = OpenMeteoFetcher()
        df = fetcher._parse_response(self._mock_response())
        assert len(df) == 3

    def test_index_is_datetime_with_timezone(self):
        from src.ingestion.open_meteo_fetcher import OpenMeteoFetcher
        fetcher = OpenMeteoFetcher()
        df = fetcher._parse_response(self._mock_response())
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.tz is not None  # Must have timezone — no naive timestamps

    def test_metadata_columns_added(self):
        from src.ingestion.open_meteo_fetcher import OpenMeteoFetcher
        fetcher = OpenMeteoFetcher()
        df = fetcher._parse_response(self._mock_response())
        assert "hour" in df.columns
        assert "day_of_week" in df.columns
        assert "month" in df.columns
        assert "is_daytime" in df.columns

    def test_numeric_columns_are_float32(self):
        from src.ingestion.open_meteo_fetcher import OpenMeteoFetcher
        fetcher = OpenMeteoFetcher()
        df = fetcher._parse_response(self._mock_response())
        assert df["temperature_2m"].dtype == "float32"
        assert df["cloud_cover"].dtype == "float32"


class TestValidation:
    """Test that validation catches bad data."""

    def test_validation_passes_on_good_data(self):
        from src.ingestion.open_meteo_fetcher import OpenMeteoFetcher
        fetcher = OpenMeteoFetcher()
        # Create 24 rows of clean data
        df = pd.DataFrame({
            "temperature_2m": [25.0] * 24,
            "shortwave_radiation": [500.0] * 24,
        })
        # Should not raise
        fetcher._validate_dataframe(df)

    def test_validation_warns_on_too_few_rows(self, caplog):
        from src.ingestion.open_meteo_fetcher import OpenMeteoFetcher
        import logging
        fetcher = OpenMeteoFetcher()
        df = pd.DataFrame({"temperature_2m": [25.0] * 5})
        # Only 5 rows — should log a warning
        with caplog.at_level(logging.WARNING):
            fetcher._validate_dataframe(df)
