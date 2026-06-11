"""
jaipur_data_simulator.py
------------------------
Generates physically realistic solar + weather data for Jaipur, Rajasthan.

WHY THIS EXISTS:
  In early development, you may not have access to live APIs or real
  plant data. Rather than using a random Kaggle dataset, we simulate
  data that matches Jaipur's actual climate profile:

  - Summer peak irradiance: ~950 W/m² (May-June)
  - Monsoon cloud cover: 60-90% (July-September)
  - Winter irradiance: ~600-700 W/m² (December-January)
  - Daily temperature range: 10-48°C across seasons

  This means models trained on simulated data will transfer reasonably
  well when real data arrives — unlike random synthetic data.

PHYSICS USED:
  - Clear-sky irradiance follows a solar angle curve (sin of elevation)
  - Cloud cover attenuates irradiance non-linearly (Beer-Lambert law)
  - Temperature follows a diurnal cycle with seasonal offset
  - Solar power output uses the standard PV equation from config
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def generate_jaipur_weather(
    start_date: str = "2024-01-01",
    days: int = 90,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate realistic hourly weather + solar data for Jaipur.

    Parameters
    ----------
    start_date : str
        Start date in "YYYY-MM-DD" format
    days : int
        Number of days to generate
    seed : int
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        Hourly DataFrame with same schema as OpenMeteoFetcher output
    """
    np.random.seed(seed)
    logger.info(f"Generating {days} days of simulated Jaipur data from {start_date}")

    # Build hourly timestamp index in IST
    start = pd.Timestamp(start_date, tz="Asia/Kolkata")
    index = pd.date_range(start=start, periods=days * 24, freq="h")
    n     = len(index)

    hours  = index.hour
    months = index.month

    # ── Solar angle & clear-sky irradiance ──────────────────────────────────
    # Simplified solar elevation: peaks at solar noon (hour 12-13 in Jaipur)
    # Jaipur latitude 26.9° — solar angle factor varies by month
    solar_noon = 12.5  # Approximate solar noon IST for Jaipur

    # Seasonal declination effect on peak irradiance
    # Summer (May=5): max ~950 W/m², Winter (Dec=12): max ~650 W/m²
    seasonal_peak = 650 + 300 * np.sin(np.pi * (months - 1) / 12)

    # Diurnal irradiance curve (positive half of sine = daylight hours)
    hour_angle    = (hours - solar_noon) * (np.pi / 12)
    solar_factor  = np.maximum(0, np.cos(hour_angle))
    clear_sky_ghi = seasonal_peak * solar_factor

    # ── Cloud cover: seasonal pattern + noise ───────────────────────────────
    # Jaipur monsoon: July (7), August (8), September (9)
    monsoon_mask  = (months >= 7) & (months <= 9)
    base_cloud    = np.where(monsoon_mask, 65.0, 15.0)  # % cloud cover
    cloud_cover   = np.clip(
        base_cloud + np.random.normal(0, 12, n),
        0, 100
    )

    # ── Irradiance attenuated by clouds ─────────────────────────────────────
    # Beer-Lambert approximation: transmission = (1 - 0.75 * (cloud/100)^3.4)
    cloud_transmission = 1 - 0.75 * (cloud_cover / 100) ** 3.4
    ghi                = clear_sky_ghi * cloud_transmission
    ghi               += np.random.normal(0, 8, n) * solar_factor  # Sensor noise
    ghi               = np.clip(ghi, 0, 1100)

    # Direct and diffuse components
    dni     = ghi * (1 - 0.3 * cloud_cover / 100) * 0.85
    diffuse = ghi * (0.3 + 0.4 * cloud_cover / 100)

    # ── Temperature: diurnal + seasonal ─────────────────────────────────────
    # Jaipur annual range: ~8°C (Jan) to ~45°C (May/June)
    seasonal_temp_mean = 18 + 22 * np.sin(np.pi * (months - 1) / 12)
    diurnal_variation  = 8 * np.sin(np.pi * (hours - 6) / 12)
    diurnal_variation  = np.where(hours < 6, -4.0, diurnal_variation)
    temperature        = seasonal_temp_mean + diurnal_variation
    temperature       += np.random.normal(0, 1.5, n)

    # ── Humidity: inverse to temperature, higher in monsoon ─────────────────
    base_humidity   = np.where(monsoon_mask, 72.0, 35.0)
    humidity        = np.clip(
        base_humidity - 0.5 * (temperature - 25) + np.random.normal(0, 5, n),
        15, 98
    )

    # ── Wind speed ───────────────────────────────────────────────────────────
    wind_speed = np.clip(
        3.5 + np.random.exponential(1.5, n),
        0, 20
    )

    # ── Simulated solar power output (PV physics) ────────────────────────────
    # P = GHI × Area × efficiency × (1 + temp_coeff × (T - 25))
    panel_efficiency  = 0.20
    temp_coefficient  = -0.004
    performance_ratio = 0.80
    capacity_mw       = 100.0
    # Normalised output: 0 to 1 relative to capacity
    temp_factor  = 1 + temp_coefficient * (temperature - 25)
    raw_output   = ghi / 1000 * temp_factor * performance_ratio
    solar_output = np.clip(raw_output * capacity_mw, 0, capacity_mw)
    solar_output += np.random.normal(0, 0.5, n) * (solar_output > 0) 
    solar_output = np.clip(solar_output, 0, capacity_mw)  # Sensor noise

    # ── Assemble DataFrame ───────────────────────────────────────────────────
    df = pd.DataFrame({
        "temperature_2m"        : temperature.astype("float32"),
        "relative_humidity_2m"  : humidity.astype("float32"),
        "precipitation"         : np.where(
                                      monsoon_mask & (cloud_cover > 70),
                                      np.random.exponential(2, n), 0
                                  ).astype("float32"),
        "cloud_cover"           : cloud_cover.astype("float32"),
        "wind_speed_10m"        : wind_speed.astype("float32"),
        "shortwave_radiation"   : ghi.astype("float32"),
        "direct_radiation"      : dni.astype("float32"),
        "diffuse_radiation"     : diffuse.astype("float32"),
        "solar_output_mw"       : solar_output.astype("float32"),  # Target variable
        "clear_sky_ghi"         : clear_sky_ghi.astype("float32"), # For clear-sky ratio feature
        "hour"                  : hours.astype("int8"),
        "day_of_week"           : index.dayofweek.astype("int8"),
        "month"                 : months.astype("int8"),
        "is_daytime"            : ((hours >= 6) & (hours <= 19)),
    }, index=index)

    logger.success(
        f"Generated {len(df)} rows | "
        f"Solar output range: {df['solar_output_mw'].min():.1f} – "
        f"{df['solar_output_mw'].max():.1f} MW | "
        f"Max irradiance: {df['shortwave_radiation'].max():.0f} W/m²"
    )
    return df


if __name__ == "__main__":
    df = generate_jaipur_weather(start_date="2024-01-01", days=90)

    # Save to processed data
    out_path = PROJECT_ROOT / "data" / "processed" / "jaipur_simulated_90d.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    logger.info(f"Saved to {out_path}")

    print("\n" + "="*60)
    print("SIMULATED JAIPUR SOLAR DATA — SUMMARY")
    print("="*60)
    print(f"\nShape: {df.shape}")
    print(f"Date range: {df.index.min()} → {df.index.max()}")
    print(f"\nKey statistics:")
    print(df[["temperature_2m","cloud_cover","shortwave_radiation","solar_output_mw"]].describe().round(2))

    print(f"\nMonthly solar output (mean MW):")
    monthly = df.groupby("month")["solar_output_mw"].mean().round(2)
    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr"}
    for m, v in monthly.items():
        bar = "█" * int(v / 3)
        print(f"  Month {m:2d}: {bar:<30} {v:.1f} MW")
