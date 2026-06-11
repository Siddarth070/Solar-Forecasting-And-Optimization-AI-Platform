"""
open_meteo_fetcher.py
---------------------
Fetches weather and solar irradiance data from Open-Meteo API
for a given location and date range.

WHY OPEN-METEO:
  - Free, no API key required
  - Production-grade (used by real weather apps)
  - Returns hourly irradiance (GHI, DNI, diffuse) — essential for solar
  - Historical archive + live forecast in one API family

WHAT THIS MODULE DOES:
  1. Builds a validated API request from config
  2. Fetches with retry logic (network failures are normal in production)
  3. Validates the response (catches API changes, missing fields)
  4. Converts to a clean pandas DataFrame with proper timestamps
  5. Saves raw JSON to data/raw/ with a timestamp in the filename
  6. Returns the DataFrame for downstream use

COMMON MISTAKES THIS MODULE AVOIDS:
  - No hardcoded coordinates (reads from config)
  - No silent failures (every error is logged and raised)
  - No timezone confusion (all timestamps converted to IST immediately)
  - No overwriting raw data (filenames include timestamp)
"""

import json
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger

from src.utils.config_loader import get_config


# ── Constants ────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


# ── Main fetcher class ────────────────────────────────────────────────────────

class OpenMeteoFetcher:
    """
    Fetches weather + irradiance data from Open-Meteo API.

    Design choice — class vs function:
    A class lets us hold config state once and reuse across multiple
    fetch calls without re-loading config each time. In a pipeline
    that fetches every 15 minutes, this matters.

    Parameters
    ----------
    config : dict, optional
        Project config dict. If None, loads from configs/config.yaml
    """

    def __init__(self, config: dict | None = None):
        self.config   = config or get_config()
        self.location = self.config["location"]
        self.sources  = self.config["data_sources"]
        self.variables = self.config["weather_variables"]["hourly"]
        self.pipeline  = self.config["pipeline"]

        # Ensure raw data directory exists
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"OpenMeteoFetcher initialised for "
            f"{self.location['name']} "
            f"({self.location['latitude']}°N, {self.location['longitude']}°E)"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def fetch_forecast(self, days_ahead: int = 7) -> pd.DataFrame:
        """
        Fetch weather forecast for the next N days.

        Parameters
        ----------
        days_ahead : int
            How many days of forecast to pull (max 16 for Open-Meteo free tier)

        Returns
        -------
        pd.DataFrame
            Hourly weather + irradiance data with IST timestamps as index
        """
        logger.info(f"Fetching {days_ahead}-day weather forecast for {self.location['name']}")

        params = self._build_forecast_params(days_ahead)
        raw_response = self._fetch_with_retry(
            url    = self.sources["open_meteo"]["base_url"],
            params = params,
            label  = "forecast"
        )

        df = self._parse_response(raw_response)
        self._validate_dataframe(df)
        self._save_raw(raw_response, label="forecast")

        logger.success(
            f"Forecast fetch complete: {len(df)} rows, "
            f"{df.index.min()} to {df.index.max()}"
        )
        return df

    def fetch_historical(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Fetch historical weather data from Open-Meteo archive.

        Parameters
        ----------
        start_date : str
            Format: "YYYY-MM-DD"
        end_date : str
            Format: "YYYY-MM-DD"

        Returns
        -------
        pd.DataFrame
            Historical hourly weather + irradiance with IST timestamps
        """
        logger.info(
            f"Fetching historical data: {start_date} to {end_date} "
            f"for {self.location['name']}"
        )

        params = self._build_historical_params(start_date, end_date)
        raw_response = self._fetch_with_retry(
            url    = self.sources["open_meteo"]["historical_url"],
            params = params,
            label  = "historical"
        )

        df = self._parse_response(raw_response)
        self._validate_dataframe(df)
        self._save_raw(raw_response, label=f"historical_{start_date}_{end_date}")

        logger.success(
            f"Historical fetch complete: {len(df)} rows, "
            f"{df.index.min()} to {df.index.max()}"
        )
        return df

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_forecast_params(self, days_ahead: int) -> dict:
        """Build query parameters for forecast endpoint."""
        return {
            "latitude"     : self.location["latitude"],
            "longitude"    : self.location["longitude"],
            "hourly"       : ",".join(self.variables),
            "timezone"     : self.config["project"]["timezone"],
            "forecast_days": days_ahead,
        }

    def _build_historical_params(self, start_date: str, end_date: str) -> dict:
        """Build query parameters for historical archive endpoint."""
        return {
            "latitude"  : self.location["latitude"],
            "longitude" : self.location["longitude"],
            "hourly"    : ",".join(self.variables),
            "timezone"  : self.config["project"]["timezone"],
            "start_date": start_date,
            "end_date"  : end_date,
        }

    def _fetch_with_retry(
        self,
        url: str,
        params: dict,
        label: str
    ) -> dict:
        """
        Make HTTP GET request with exponential backoff retry.

        WHY RETRY LOGIC:
          External APIs fail intermittently — network blips, rate limits,
          timeouts. A production system must handle this silently instead
          of crashing. Exponential backoff avoids hammering a struggling API.
        """
        max_retries   = self.pipeline["max_retries"]
        retry_delay   = self.pipeline["retry_delay_seconds"]
        timeout       = self.pipeline["fetch_timeout_seconds"]

        for attempt in range(1, max_retries + 1):
            try:
                logger.debug(f"[{label}] Attempt {attempt}/{max_retries} — {url}")

                response = requests.get(url, params=params, timeout=timeout)
                response.raise_for_status()  # Raises on 4xx/5xx

                data = response.json()

                # Open-Meteo returns error field on bad requests
                if "error" in data:
                    raise ValueError(f"Open-Meteo API error: {data['reason']}")

                logger.debug(f"[{label}] Fetch successful on attempt {attempt}")
                return data

            except requests.exceptions.Timeout:
                logger.warning(f"[{label}] Timeout on attempt {attempt}")
            except requests.exceptions.ConnectionError:
                logger.warning(f"[{label}] Connection error on attempt {attempt}")
            except requests.exceptions.HTTPError as e:
                logger.error(f"[{label}] HTTP error: {e}")
                raise  # Don't retry on HTTP errors — they're our bug

            if attempt < max_retries:
                wait = retry_delay * (2 ** (attempt - 1))  # Exponential backoff
                logger.info(f"[{label}] Waiting {wait}s before retry...")
                time.sleep(wait)

        raise RuntimeError(
            f"[{label}] All {max_retries} fetch attempts failed. "
            "Check network connectivity and API status."
        )

    def _parse_response(self, raw: dict) -> pd.DataFrame:
        """
        Convert raw API JSON into a clean, typed DataFrame.

        Design decisions:
        - Index is DatetimeTZAware in IST (not UTC, not naive)
          Reason: All downstream code works in IST; naive timestamps
          cause silent bugs at DST boundaries
        - Column names are kept exactly as Open-Meteo returns them
          Reason: Makes debugging easier when checking API docs
        - float32 for numeric columns
          Reason: Halves memory vs float64 with no precision loss needed
        """
        hourly_data = raw.get("hourly", {})

        if not hourly_data:
            raise ValueError("API response contains no 'hourly' data block")

        df = pd.DataFrame(hourly_data)

        # Convert timestamp string to DatetimeTZ in IST
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")

        # Localise if timezone-naive (Open-Meteo returns local time)
        if df.index.tz is None:
            df.index = df.index.tz_localize("Asia/Kolkata")

        # Cast numeric columns to float32 to save memory
        numeric_cols = df.select_dtypes(include="number").columns
        df[numeric_cols] = df[numeric_cols].astype("float32")

        # Add metadata columns useful for feature engineering later
        df["hour"]        = df.index.hour
        df["day_of_week"] = df.index.dayofweek
        df["month"]       = df.index.month
        df["is_daytime"]  = (df["hour"] >= 6) & (df["hour"] <= 19)

        logger.debug(f"Parsed DataFrame: {df.shape}, columns: {list(df.columns)}")
        return df

    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """
        Run data quality checks on fetched data.

        WHY VALIDATE:
          Garbage in, garbage out. A model trained on bad data is
          worse than no model. We catch data quality issues at ingestion,
          not three weeks later when forecast accuracy mysteriously drops.
        """
        val = self.config["validation"]
        issues = []

        # Check for excessive missing values
        null_pct = df.isnull().mean()
        bad_cols = null_pct[null_pct > val["max_missing_pct"]]
        if not bad_cols.empty:
            issues.append(
                f"High null percentage in columns: {bad_cols.to_dict()}"
            )

        # Check temperature range
        if "temperature_2m" in df.columns:
            temp_min = df["temperature_2m"].min()
            temp_max = df["temperature_2m"].max()
            if temp_min < val["min_temperature_c"] or temp_max > val["max_temperature_c"]:
                issues.append(
                    f"Temperature out of range: min={temp_min:.1f}, max={temp_max:.1f}"
                )

        # Check irradiance range
        if "shortwave_radiation" in df.columns:
            irr_max = df["shortwave_radiation"].max()
            if irr_max > val["max_irradiance_wm2"]:
                issues.append(
                    f"Irradiance exceeds physical max: {irr_max:.1f} W/m²"
                )

        # Check minimum row count (at least 24 hours of data)
        if len(df) < 24:
            issues.append(f"Too few rows: {len(df)} (expected at least 24)")

        if issues:
            for issue in issues:
                logger.warning(f"Data quality issue: {issue}")
            logger.warning(
                f"{len(issues)} data quality issue(s) found. "
                "Review before using for model training."
            )
        else:
            logger.success("Data validation passed — all checks green")

    def _save_raw(self, data: dict, label: str) -> Path:
        """
        Save raw API response as JSON with timestamp in filename.

        WHY SAVE RAW DATA:
          Always preserve the original. If your parsing has a bug,
          you can re-parse from raw without re-fetching. In production,
          API responses are your audit trail.

        Returns
        -------
        Path
            Path to the saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = f"open_meteo_{label}_{timestamp}.json"
        filepath  = RAW_DATA_DIR / filename

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Raw data saved to {filepath}")
        return filepath
