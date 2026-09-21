"""
synthetic_forecast.py — A stand-in for a real day-ahead weather forecast.
----------------------------------------------------------------------------

WHY THIS EXISTS:
  Every weather column the simulator (src/ingestion/jaipur_simulator.py)
  produces is the TRUE, already-happened value -- shortwave_radiation is
  what the sun actually did, not what a forecast issued the day before
  would have said. A real deployment never gets the true value ahead of
  time; it gets a D-1 09:00 NWP forecast, which is wrong by a
  meaningful, physically realistic amount.

  Training a model on true weather and then serving it forecast weather
  is a distribution mismatch the model was never taught to handle (see
  benchmark.py's P0.5 "deliverable skill" scoring, which caught exactly
  this: the model looked excellent fed true weather and collapsed below
  a trivial baseline fed realistic forecast weather).

CAVEAT: there is no real day-ahead forecast archive in this repo yet
(Open-Meteo's archive API returns observed history, not what would have
been forecast the day before) -- acquiring one is a real-data task for a
later phase. This is a clearly-labeled synthetic proxy so training and
evaluation can both use forecast-QUALITY inputs before that data exists.
Replace this module, not its callers, once a real forecast archive
exists.
"""

import numpy as np
import pandas as pd

# ~15% relative error on irradiance is a realistic day-ahead NWP ballpark.
GHI_RELATIVE_NOISE_STD = 0.15
CLOUD_COVER_NOISE_STD = 12.0


def simulate_day_ahead_forecast(raw: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Return a copy of `raw` with shortwave_radiation and cloud_cover
    perturbed to look like a D-1 09:00 forecast instead of the true,
    already-happened value. Every other column (including
    solar_output_mw, if present) is left untouched -- what actually
    generated is real; only the weather INPUT a forecaster would have had
    ahead of time is degraded."""
    rng = np.random.default_rng(seed)
    forecast = raw.copy()
    n = len(forecast)

    ghi_noise = 1 + rng.normal(0, GHI_RELATIVE_NOISE_STD, n)
    forecast["shortwave_radiation"] = (forecast["shortwave_radiation"] * ghi_noise).clip(lower=0)

    cloud_noise = rng.normal(0, CLOUD_COVER_NOISE_STD, n)
    forecast["cloud_cover"] = (forecast["cloud_cover"] + cloud_noise).clip(0, 100)

    return forecast
