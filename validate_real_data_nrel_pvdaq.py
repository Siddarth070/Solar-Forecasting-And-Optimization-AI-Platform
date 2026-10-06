"""
validate_real_data_nrel_pvdaq.py
-----------------------------------------------------------------------
One-off real-data sanity check: do this project's P0.4 baselines and
P2.2 quality gate produce sane results on REAL measured generation data,
the first time either has ever been run on anything but the synthetic
Jaipur/Pune simulator?

THIS IS NOT A PRODUCT PLANT AND NOT A SUBSTITUTE FOR ROADMAP P3.1:
  - It's a US system (Kersey, Colorado) -- none of this project's CERC
    DSM / IEGC grid-code compliance logic applies to it, and it proves
    nothing about an Indian plant.
  - It's a single-axis-tracker, CdTe (thin-film) plant -- this project's
    physics_baseline formula and location.yaml schema assume a
    fixed-tilt, generic-silicon plant, so the "physics" and
    "performance_ratio" numbers below are approximations, not this
    project's normal methodology applied cleanly (see inline notes).
  - Several plant facts below are INFERRED from the real data (AC
    capacity, performance_ratio), not read from a real nameplate spec,
    because the dataset's own metadata doesn't give them. Each is
    labeled as such in its own print statement -- never presented as a
    verified fact.

WHAT IT DID FIND, FOR REAL:
  Running this against system 9068 surfaced a genuine bug in
  src/evaluation/baselines.py's smart_persistence_baseline: a near-zero
  (not exactly zero) clear-sky-power denominator at dawn/dusk -- which
  this project's synthetic simulator never produces, but real sensor
  data does -- produced an unbounded ratio that blew up the next day's
  prediction (one row predicted 715 MW for a ~4 MW plant). Fixed by
  bounding the ratio to the same [0, 1.3] range this project's own
  clear_sky_index already uses elsewhere (src/features/pipeline.py).
  See tests/test_baselines.py::TestSmartPersistenceBaseline::
  test_near_zero_clear_sky_power_does_not_blow_up for the regression
  test, and git history for the before/after real-data numbers.

DATA SOURCE, LICENSE, AND HOW TO RE-RUN THIS:
  NREL PVDAQ Public Data Lake (part of the DOE Solar Data Prize 2023),
  hosted on OEDI: https://dx.doi.org/10.25984/1846021
  Copyright (c) 2024, Alliance for Sustainable Energy, LLC. Released
  under a BSD-3-Clause-style license (see
  https://raw.githubusercontent.com/openEDI/documentation/main/pvdaq.md
  for the exact terms). System 9068 ("SR_CO"), a public, anonymized
  research system -- not a Zenith customer, not India, not under any
  NDA. This repository does not vendor the raw data (it's ~380 MB);
  download it yourself before running this script:

    mkdir -p /tmp/pvdaq_9068 && cd /tmp/pvdaq_9068
    for f in ac_power_data irradiance_data environment_data; do
      curl -O "https://oedi-data-lake.s3.amazonaws.com/pvdaq/2023-solar-data-prize/9068_OEDI/data/9068_${f}.csv"
    done

  Then: python3 validate_real_data_nrel_pvdaq.py /tmp/pvdaq_9068
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.baselines import (
    nmae_nrmse,
    persistence_baseline,
    physics_baseline,
    smart_persistence_baseline,
)
from src.features.pipeline import clear_sky_ghi
from src.quality.gate import run_quality_checks

# Real facts about system 9068, from its own metadata JSON
# (pvdaq/2023-solar-data-prize/9068_OEDI/metadata/9068_system_metadata.json)
# -- Kersey, CO, single-axis tracker, CdTe, 4.738 MW DC nameplate.
LATITUDE, LONGITUDE, ELEVATION_M = 40.3864, -104.5512, 1407.0
NAMEPLATE_DC_MW = 4.738
# Typical published CdTe (First-Solar-class) figure -- this system's own
# metadata lists its module manufacturer/model as "Unknown", so this is a
# generic technology figure, not this module's real spec.
TEMPERATURE_COEFFICIENT = -0.0025


def main(data_dir: str):
    ac = pd.read_csv(f"{data_dir}/9068_ac_power_data.csv",
                      usecols=["measured_on", "meter_ac_power_(kw)_meter_150150"])
    ac["measured_on"] = pd.to_datetime(ac["measured_on"])
    ac = ac.set_index("measured_on").sort_index().rename(
        columns={"meter_ac_power_(kw)_meter_150150": "meter_kw"})

    irr = pd.read_csv(f"{data_dir}/9068_irradiance_data.csv",
                       usecols=["measured_on", "pyranometer_(class_a)_pad_1_poa_irradiance_(w/m2)_o_149723"])
    irr["measured_on"] = pd.to_datetime(irr["measured_on"])
    irr = irr.set_index("measured_on").sort_index().rename(
        columns={"pyranometer_(class_a)_pad_1_poa_irradiance_(w/m2)_o_149723": "poa_wm2"})

    env = pd.read_csv(f"{data_dir}/9068_environment_data.csv",
                       usecols=["measured_on", "weather_station_ambient_temperature_(c)_o_149727"])
    env["measured_on"] = pd.to_datetime(env["measured_on"])
    env = env.set_index("measured_on").sort_index().rename(
        columns={"weather_station_ambient_temperature_(c)_o_149727": "temp_c"})

    raw = ac.join(irr, how="inner").join(env, how="inner")
    print(f"Joined raw 5-min rows: {len(raw)}  ({raw.index.min()} to {raw.index.max()})")

    # Real data quality: a few null rows, and real negative meter readings
    # (grid-side parasitic draw at night, not generation) -- clip rather
    # than pretend negative output is meaningful or silently drop it.
    n_null = raw.isna().any(axis=1).sum()
    raw = raw.dropna()
    n_negative_meter = (raw["meter_kw"] < 0).sum()
    raw["meter_kw"] = raw["meter_kw"].clip(lower=0)
    print(f"Dropped {n_null} null row(s); clipped {n_negative_meter} negative meter reading(s) to 0.")

    # Real multi-year data crosses DST boundaries; drop the handful of
    # ambiguous/nonexistent clock times rather than guess which side they're on.
    before = len(raw)
    raw.index = raw.index.tz_localize("America/Denver", ambiguous="NaT", nonexistent="NaT")
    raw = raw[raw.index.notna()]
    print(f"Dropped {before - len(raw)} row(s) at DST ambiguous/nonexistent clock times.")

    # Resample to hourly: src.evaluation.baselines' LAG_HOURS=24 assumes an
    # hourly cadence; 5-min data fed directly would shift by 2 hours, not 24.
    hourly = raw.resample("1h").mean()
    hourly["hour"] = hourly.index.hour
    hourly["month"] = hourly.index.month
    hourly["solar_output_mw"] = hourly["meter_kw"] / 1000.0
    # This sensor measures plane-of-array (POA) irradiance, not GHI -- for
    # this site's single-axis tracker, POA is the more physically relevant
    # measured quantity anyway (a tracker's output doesn't relate to flat
    # GHI the simple way this project's physics_baseline formula assumes
    # for a FIXED-TILT plant). Reusing this project's "shortwave_radiation"/
    # "temperature_2m" column names lets src.evaluation.baselines' functions
    # run unmodified; treat "physics" baseline numbers below as approximate
    # for that reason.
    hourly["shortwave_radiation"] = hourly["poa_wm2"]
    hourly["temperature_2m"] = hourly["temp_c"]
    hourly = hourly.dropna(subset=["solar_output_mw", "shortwave_radiation", "temperature_2m"])
    print(f"Hourly rows after resample: {len(hourly)}")

    ac_capacity_mw = float(hourly["solar_output_mw"].quantile(0.995))
    print(f"\nInferred AC capacity (99.5th percentile of real hourly output, "
          f"NOT a nameplate spec): {ac_capacity_mw:.3f} MW  "
          f"(nameplate DC per metadata: {NAMEPLATE_DC_MW} MW -- ratio "
          f"{ac_capacity_mw / NAMEPLATE_DC_MW:.2f}, a plausible real AC/DC derate)")

    csghi = clear_sky_ghi(hourly.index, LATITUDE, LONGITUDE, ELEVATION_M)
    hourly["clear_sky_ghi_model"] = csghi.to_numpy()
    daytime = hourly["clear_sky_ghi_model"] > 1.0

    plant_config = {
        "plant_id": "research_pvdaq_9068_sr_co",
        "location": {"latitude": LATITUDE, "longitude": LONGITUDE, "elevation_m": ELEVATION_M},
        "capacity": {
            "ac_capacity_mw": ac_capacity_mw,
            "performance_ratio": 1.0,  # placeholder, fit below
            "temperature_coefficient": TEMPERATURE_COEFFICIENT,
        },
    }
    # Fit performance_ratio by least squares against real daytime data --
    # physics_baseline's GHI-based, fixed-tilt formula doesn't map cleanly
    # onto a POA-measured, single-axis-tracker system, so an assumed
    # "nameplate" PR would be fabricated for this plant; a fitted scale
    # factor is honest about what it actually is.
    day = hourly[daytime]
    temp_factor = 1 + TEMPERATURE_COEFFICIENT * (day["temperature_2m"] - 25)
    raw_frac = (day["shortwave_radiation"] / 1000 * temp_factor).clip(lower=0)
    denom = float((raw_frac ** 2).sum())
    numer = float((raw_frac * (day["solar_output_mw"] / ac_capacity_mw)).sum())
    fitted_pr = numer / denom if denom > 0 else float("nan")
    plant_config["capacity"]["performance_ratio"] = fitted_pr
    print(f"Fitted performance_ratio (least-squares against real data, "
          f"NOT a nameplate spec): {fitted_pr:.3f}")

    print("\n--- P2.2 quality gate on real hourly data (first time ever run on non-synthetic data) ---")
    qreport = run_quality_checks(hourly, plant_config, power_column="solar_output_mw")
    print(qreport.summary())

    y = hourly["solar_output_mw"]
    persistence = persistence_baseline(y)
    smart = smart_persistence_baseline(y, hourly, plant_config)
    physics = physics_baseline(hourly, plant_config)

    print("\n--- P0.4 baselines vs. REAL actuals (whole ~6.2-year real history) ---")
    for name, preds in [("persistence", persistence), ("smart_persistence", smart), ("physics", physics)]:
        nmae, nrmse, n = nmae_nrmse(y, preds, ac_capacity_mw, daytime)
        print(f"  {name:18s}  nMAE {nmae:6.2f}%   nRMSE {nrmse:6.2f}%   (n={n})")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python3 validate_real_data_nrel_pvdaq.py <data_dir>")
    main(sys.argv[1])
