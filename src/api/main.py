import copy
import json
import os
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from loguru import logger
from xgboost import XGBRegressor

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.attribution.loss import attribute_losses, daily_performance_ratio, detect_soiling
from src.features.pipeline import build_features, SERVING_FEATURE_COLUMNS
from src.ingestion.open_meteo_fetcher import OpenMeteoFetcher
from src.onboarding.plant_registration import PlantAlreadyRegisteredError, register_plant
from src.optimization.battery_optimizer import BatteryOptimizer
from src.quality.gate import run_quality_checks
from src.recommendations import store as recommendations_store
from src.recommendations.engine import (
    recommend_battery_actions,
    recommend_inspections,
    recommend_schedule_revisions,
)
from src.regulatory import dsm, grid_code
from src.reporting.weekly_report import generate_weekly_report
from src.risk.schedule_risk import score_schedule_risk
from src.scheduling.rolling_horizon import apply_schedule_revision
from src.time_blocks import BLOCKS_PER_DAY, BLOCK_MINUTES, block_boundaries, integrate_to_blocks
from src.utils.config_loader import get_plant_config, list_plant_ids

DEFAULT_PLANT_ID = "jaipur_100mw"


def get_recommendations_db():
    """A fresh connection to the recommendation log (roadmap P2.6) on
    every call -- SQLite handles this cheaply, and it keeps request
    handling stateless. Tests override this (see
    tests/test_recommendations_api.py) to point at one shared in-memory
    connection instead of the real on-disk log."""
    return recommendations_store.connect()


def build_timestamp_index(raw_timestamps: list[str]) -> pd.DatetimeIndex:
    """Parse ISO timestamp strings into one consistent, homogeneous
    DatetimeIndex. Raises ValueError -- meant to be caught and turned
    into a 422 -- if they can't form one, e.g. a mix of tz-aware and
    tz-naive timestamps (pandas can't unify those into a single
    DatetimeIndex at all, and would otherwise surface as a confusing
    internal pandas error). Endpoints that need already-clean timestamps
    (everything except POST /quality/check, whose entire job is
    diagnosing exactly this kind of raw-data problem) should use this
    instead of constructing a DatetimeIndex directly."""
    idx = pd.Index([pd.Timestamp(t) for t in raw_timestamps])
    if not isinstance(idx, pd.DatetimeIndex):
        raise ValueError(
            "Timestamps could not be parsed into one consistent timezone -- e.g. some "
            "carry a UTC offset and others don't. Use POST /quality/check first to "
            "diagnose raw data like this."
        )
    return idx


# ── App setup ─────────────────────────────────────────────────
app = FastAPI(
    title="Solar Forecast Platform",
    description="AI-based solar energy forecasting and grid optimization, "
                 "for any plant configured under configs/plants/",
    version="1.0.0"
)

# ── CORS (web console, web/) ──────────────────────────────────
# The web console is a separate static site, so the browser needs an
# explicit allow-list -- never "*". Set ZENITH_CORS_ORIGINS (comma-
# separated) in every deployment; the defaults cover local dev only.
_DEFAULT_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8080"
CORS_ALLOW_ORIGINS = [
    o.strip() for o in os.getenv("ZENITH_CORS_ORIGINS", _DEFAULT_ORIGINS).split(",") if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)

# ── Load model on startup ─────────────────────────────────────
# JSON, not pickle: a pickle can silently break across library versions
# and is opaque to review (roadmap P0.6).
MODEL_PATH = PROJECT_ROOT / "src" / "models" / "xgboost_solar_v2.json"
QUANTILE_MODEL_PATH = PROJECT_ROOT / "src" / "models" / "xgboost_solar_quantile.json"
MODEL_CARD_PATH = PROJECT_ROOT / "src" / "models" / "model_card.json"

if MODEL_PATH.exists():
    model = XGBRegressor()
    model.load_model(str(MODEL_PATH))
    logger.success(f"Model loaded from {MODEL_PATH}")
else:
    logger.error(f"Model not found at {MODEL_PATH}")
    model = None

# Probabilistic P10/P50/P90 model (roadmap P1.3) -- optional: the point
# model above remains fully functional without it, so its absence is
# logged, not fatal.
if QUANTILE_MODEL_PATH.exists():
    quantile_model = XGBRegressor()
    quantile_model.load_model(str(QUANTILE_MODEL_PATH))
    logger.success(f"Quantile model loaded from {QUANTILE_MODEL_PATH}")
else:
    logger.warning(f"Quantile model not found at {QUANTILE_MODEL_PATH} -- "
                    f"/forecast will omit predictions_p10/p50/p90_mw")
    quantile_model = None


# ── Request/Response schemas ──────────────────────────────────

class WeatherInput(BaseModel):
    """Input weather data for one real timestamp (roadmap P1.2/P1.4's
    96-block grid: a genuine ISO timestamp, not a hour-of-day/month pair
    anchored to a fake placeholder date)."""
    timestamp             : str   = Field(..., description="ISO8601 timestamp for this "
                                           "reading, e.g. '2024-06-01T09:00:00+05:30'.")
    shortwave_radiation   : float = Field(..., ge=0, le=1200, description="GHI in W/m²")
    cloud_cover           : float = Field(..., ge=0, le=100,  description="Cloud cover %")
    temperature_2m        : float = Field(..., ge=-10, le=60, description="Temperature °C")
    relative_humidity_2m  : float = Field(..., ge=0, le=100,  description="Humidity %")
    wind_speed_10m        : float = Field(..., ge=0, le=50,   description="Wind speed m/s")


class ForecastRequest(BaseModel):
    """Request body for forecast endpoint."""
    hours: list[WeatherInput] = Field(
        ...,
        min_length=1,
        max_length=168,
        description="List of hourly weather inputs"
    )
    plant_id: str = Field(
        default=DEFAULT_PLANT_ID,
        description="Which configured plant to forecast for — matches a "
                     "file under configs/plants/. See GET /plants."
    )
    plant_capacity_mw: float | None = Field(
        default=None,
        description="Override the plant's configured AC capacity (MW). "
                     "Defaults to the plant's own configured capacity."
    )


class ForecastResponse(BaseModel):
    """Response from forecast endpoint."""
    forecast_hours     : int
    timestamps         : list[str] = Field(
        description="The real timestamps each prediction corresponds to, "
                     "echoing request.hours[i].timestamp in order."
    )
    predictions_mw     : list[float]
    peak_output_mw     : float
    peak_index         : int = Field(description="Position in predictions_mw/timestamps of the peak.")
    peak_timestamp     : str
    total_generation_mwh: float = Field(
        description="Energy over the request: sum(MW) x interval_hours. "
                     "Correct for hourly AND 15-minute inputs."
    )
    interval_hours     : float = Field(
        description="Spacing between the request's timestamps, in hours "
                     "(0.25 for a 96-block day, 1.0 for hourly)."
    )
    generated_at       : str
    predictions_p10_mw : list[float] = Field(
        default_factory=list,
        description="10th-percentile forecast per hour (roadmap P1.3). "
                     "Empty if the quantile model isn't loaded."
    )
    predictions_p50_mw : list[float] = Field(
        default_factory=list,
        description="Median (50th-percentile) forecast per hour, from the "
                     "quantile model -- not identical to predictions_mw, "
                     "which comes from the separately-trained point model."
    )
    predictions_p90_mw : list[float] = Field(
        default_factory=list,
        description="90th-percentile forecast per hour (roadmap P1.3)."
    )


class OptimizeRequest(BaseModel):
    """Request body for optimization endpoint."""
    solar_forecast_mw : list[float] = Field(..., description="Solar forecast, one value per block")
    declared_schedule_mw: list[float] = Field(
        ..., description="What the plant committed to deliver to the grid, "
                          "one value per block — not a demand forecast "
                          "(roadmap P1.6: a solar IPP has a schedule, not demand)."
    )
    battery_capacity_mwh : float = Field(default=50.0)
    charge_rate_mw       : float = Field(default=25.0)
    discharge_rate_mw    : float = Field(default=25.0)
    initial_charge_mwh   : float = Field(default=25.0)
    dt_hours              : float = Field(default=0.25, description="Block duration in hours")
    round_trip_efficiency : float = Field(default=0.90)
    plant_id              : str | None = Field(
        default=None,
        description="If given, activates real CERC DSM-charge-based "
                     "optimization (roadmap P1.7) using this plant's "
                     "configured regulatory.seller_category and "
                     "regulatory.contract_rate_rs_per_kwh, in place of a "
                     "flat deviation penalty. See GET /plants."
    )
    date                  : str | None = Field(
        default=None,
        description="Calendar date (YYYY-MM-DD) this schedule covers "
                     f"(roadmap P1.4). If given, solar_forecast_mw and "
                     f"declared_schedule_mw MUST have exactly {BLOCKS_PER_DAY} "
                     "entries -- one per real 15-minute block of that date "
                     "(src/time_blocks.py) -- dt_hours is fixed at 0.25 to "
                     "match, and the response's schedule rows carry real "
                     "block-start timestamps."
    )


class ReviseScheduleRequest(BaseModel):
    """Request body for the schedule-revision endpoint (roadmap P1.4)."""
    plant_id: str = Field(default=DEFAULT_PLANT_ID)
    date: str = Field(..., description="Calendar date (YYYY-MM-DD) the schedule covers.")
    locked_schedule_mw: list[float] = Field(
        ..., min_length=BLOCKS_PER_DAY, max_length=BLOCKS_PER_DAY,
        description=f"The currently-declared schedule, exactly {BLOCKS_PER_DAY} values "
                     "(one per 15-minute block, block 1 = 00:00-00:15)."
    )
    proposed_schedule_mw: list[float] = Field(
        ..., min_length=BLOCKS_PER_DAY, max_length=BLOCKS_PER_DAY,
        description="A candidate revised schedule (e.g. from an updated intraday "
                     "forecast), same length and block alignment as locked_schedule_mw."
    )
    request_timestamp: str = Field(
        ..., description="ISO timestamp of when this revision is being requested "
                          "(real time 'now'), e.g. '2024-06-01T13:56:00+05:30'."
    )


class ReviseScheduleResponse(BaseModel):
    """Response from the schedule-revision endpoint."""
    applied_schedule_mw: list[float]
    effective_timestamp: str | None = Field(
        description="When the revision actually takes effect per CERC IEGC 2023 "
                     "Regulation 49(4)(c) -- null if the revision wasn't allowed at all."
    )
    revision_allowed: bool = Field(
        description="False if this plant's configured transaction_type isn't "
                     "'bilateral' (Regulation 49(8) permits revision only for "
                     "bilateral WS-seller transactions, not collective)."
    )
    locked_block_count: int = Field(
        description="How many of the 96 blocks stayed locked at their old value."
    )


class LiveForecastResponse(BaseModel):
    """Response from GET /forecast/live -- one plant-day on the real
    96-block grid, from server-fetched Open-Meteo weather."""
    plant_id            : str
    date                : str
    capacity_mw         : float
    block_starts        : list[str] = Field(description=f"{BLOCKS_PER_DAY} block-start timestamps (IST).")
    predictions_mw      : list[float] = Field(description="Point forecast, mean MW per block.")
    predictions_p10_mw  : list[float] = Field(default_factory=list)
    predictions_p50_mw  : list[float] = Field(default_factory=list)
    predictions_p90_mw  : list[float] = Field(default_factory=list)
    peak_output_mw      : float
    peak_block_start    : str
    total_generation_mwh: float
    weather_hourly      : dict = Field(
        description="The hourly Open-Meteo inputs the forecast was made from: "
                     "timestamps plus one list per weather variable."
    )
    weather_source      : str
    generated_at        : str


class DsmEstimateRequest(BaseModel):
    """Request body for POST /dsm/estimate (CERC DSM Regulations 2024,
    Regulation 8(4), via src/regulatory/dsm.py)."""
    plant_id: str = Field(default=DEFAULT_PLANT_ID)
    scheduled_mw: list[float] = Field(..., min_length=1, description="Declared schedule, mean MW per block.")
    injected_mw: list[float] = Field(
        ..., min_length=1,
        description="Actual (or forecast/dispatched) injection, mean MW per block, "
                     "same length and alignment as scheduled_mw."
    )
    dt_hours: float = Field(default=0.25, gt=0, le=1, description="Block duration in hours.")
    date: str | None = Field(
        default=None,
        description=f"YYYY-MM-DD. If given, both lists must have exactly {BLOCKS_PER_DAY} "
                     "entries, dt_hours is fixed at 0.25, rows carry block-start timestamps, "
                     "and the date selects the pre/post-01.04.2026 volume limits."
    )


# ── Helpers ───────────────────────────────────────────────────

def _load_plant(plant_id: str) -> dict:
    try:
        return get_plant_config(plant_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


def _parse_date(value: str) -> pd.Timestamp:
    try:
        return pd.Timestamp(datetime.strptime(value, "%Y-%m-%d"))
    except ValueError:
        raise HTTPException(status_code=422, detail=f"date must be YYYY-MM-DD, got {value!r}")


def _predict_mw(raw: pd.DataFrame, plant_config: dict, capacity_mw: float):
    """Point forecast (MW) and, if the quantile model is loaded, an (n, 3)
    P10/P50/P90 array (MW) -- else None. `raw` is hourly weather indexed by
    real tz-aware timestamps, with hour/month columns."""
    features = build_features(raw, plant_config)
    X = features[SERVING_FEATURE_COLUMNS]
    # Model outputs capacity FRACTION (0-1), not MW (see
    # src/models/train.py) -- rescale by the requested plant's own
    # capacity so one model correctly serves differently-sized plants.
    point = np.clip(model.predict(X) * capacity_mw, 0, capacity_mw)

    q_preds = None
    if quantile_model is not None:
        # Quantile model outputs capacity FRACTION, three columns in
        # [P10, P50, P90] order -- rescale, then sort defensively so a
        # rare crossing never produces P10 > P90 (see benchmark.py).
        q_preds = np.clip(quantile_model.predict(X) * capacity_mw, 0, capacity_mw)
        q_preds = np.sort(q_preds, axis=1)
    return point, q_preds


def _fetch_weather_for_date(plant_config: dict, date: str) -> pd.DataFrame:
    """Hourly Open-Meteo forecast for `date` 00:00 through the next day's
    00:00 (IST). Module-level so tests can monkeypatch the network out."""
    return OpenMeteoFetcher(plant_config).fetch_forecast_for_date(date)


def _hourly_mw_to_blocks(values: np.ndarray, index: pd.DatetimeIndex, date: str) -> list[float]:
    """Hourly MW samples -> mean MW per real 15-minute block of `date`,
    via the same time-weighted integration src/time_blocks.py's own tests
    verify (MWh per block / block hours). For P10/P90 this is an
    approximation: a per-block quantile is taken as the linear
    interpolation of the hourly quantiles, which is not exact."""
    energy_mwh = integrate_to_blocks(pd.Series(values, index=index), date)
    return [round(float(v), 3) for v in (energy_mwh / (BLOCK_MINUTES / 60)).tolist()]


class GenerationReading(BaseModel):
    """One raw generation reading to be quality-checked (roadmap P2.2)."""
    timestamp: str = Field(..., description="ISO8601 timestamp, e.g. '2024-06-01T09:00:00+05:30'.")
    power_mw: float = Field(..., description="Reported generation in MW -- not clamped or "
                                              "validated here; that's exactly what this endpoint checks.")


class QualityCheckRequest(BaseModel):
    """Request body for the data-quality gate (roadmap P2.2)."""
    plant_id: str = Field(default=DEFAULT_PLANT_ID)
    readings: list[GenerationReading] = Field(..., min_length=1)


class LossReading(BaseModel):
    """One raw generation + weather reading for loss attribution (roadmap
    P2.4). Unlike the quality gate, attribution needs the actual measured
    irradiance and temperature too -- it classifies losses against
    expected-from-irradiance, which the plant's power reading alone can't
    reconstruct."""
    timestamp: str = Field(..., description="ISO8601 timestamp, e.g. '2024-06-01T09:00:00+05:30'.")
    power_mw: float = Field(..., description="Reported generation in MW.")
    ghi_w_m2: float = Field(..., description="Measured global horizontal irradiance, W/m^2.")
    temperature_c: float = Field(..., description="Measured ambient temperature, degC.")


class LossAttributionRequest(BaseModel):
    """Request body for the loss attribution engine (roadmap P2.4)."""
    plant_id: str = Field(default=DEFAULT_PLANT_ID)
    readings: list[LossReading] = Field(..., min_length=1)


class ScheduleRiskRequest(BaseModel):
    """Request body for the schedule-risk scoring endpoint (roadmap P2.5)."""
    plant_id: str = Field(
        default=DEFAULT_PLANT_ID,
        description="Must have regulatory.seller_category and "
                     "regulatory.contract_rate_rs_per_kwh configured -- unlike "
                     "/optimize, this endpoint has no flat-penalty fallback, "
                     "since a risk score without the real DSM settlement math "
                     "behind it would not mean anything."
    )
    declared_schedule_mw: list[float] = Field(..., min_length=1)
    p10_mw: list[float] = Field(..., min_length=1)
    p50_mw: list[float] = Field(..., min_length=1)
    p90_mw: list[float] = Field(..., min_length=1)
    dt_hours: float = Field(default=0.25, description="Block duration in hours")
    as_of: str | None = Field(
        default=None,
        description="Calendar date (YYYY-MM-DD) deciding which side of the CERC DSM "
                     "01.04.2026 cutover applies (roadmap P1.7). Defaults to `date` if "
                     "given, else today."
    )
    date: str | None = Field(
        default=None,
        description="Calendar date (YYYY-MM-DD) this schedule covers (roadmap P1.4). If "
                     f"given, all four *_mw lists MUST have exactly {BLOCKS_PER_DAY} "
                     "entries -- one per real 15-minute block of that date -- dt_hours is "
                     "fixed at 0.25, and the response carries real block-start timestamps."
    )


class BatteryStateInput(BaseModel):
    """Current battery state, for the battery-action recommendation rule
    (roadmap P2.6). Omit entirely to skip that rule (no battery
    configured for this plant) rather than guessing specs."""
    soc_mwh: float = Field(..., description="Current state of charge, MWh.")
    capacity_mwh: float = Field(..., description="Usable battery capacity, MWh.")
    charge_rate_mw: float = Field(..., description="Maximum charge rate, MW.")
    discharge_rate_mw: float = Field(..., description="Maximum discharge rate, MW.")


class ScheduleRiskInputs(BaseModel):
    """Same shape as ScheduleRiskRequest, nested here so one
    /recommendations/generate call can drive both the schedule-revision
    and battery-action rules, which both need a schedule-risk report."""
    declared_schedule_mw: list[float] = Field(..., min_length=1)
    p10_mw: list[float] = Field(..., min_length=1)
    p50_mw: list[float] = Field(..., min_length=1)
    p90_mw: list[float] = Field(..., min_length=1)
    dt_hours: float = Field(default=0.25)
    as_of: str | None = Field(default=None)
    date: str | None = Field(default=None)


class RecommendationsGenerateRequest(BaseModel):
    """Request body for the operator-recommendation engine (roadmap
    P2.6). `schedule_risk` drives the schedule-revision and
    battery-action rules; `loss_readings` drives the inspection rule.
    Either or both may be given -- omitting one just skips the rules
    that need it, rather than erroring."""
    plant_id: str = Field(default=DEFAULT_PLANT_ID)
    schedule_risk: ScheduleRiskInputs | None = Field(default=None)
    loss_readings: list[LossReading] | None = Field(default=None)
    battery_state: BatteryStateInput | None = Field(default=None)
    now: str | None = Field(
        default=None,
        description="Real 'now' timestamp, for checking whether each high-risk block's "
                     "revision gate is still open (roadmap P1.4). Without it, schedule-"
                     "revision recommendations are still produced, but say the gate wasn't "
                     "checked."
    )


class RecommendationDecisionRequest(BaseModel):
    """Request body for approving/dismissing a logged recommendation
    (roadmap P2.6) -- human-approved only, and the decision itself is
    logged, never silently applied."""
    decision: str = Field(..., description="'approved' or 'dismissed'.")
    decided_by: str = Field(..., description="Who made this decision -- required for the audit log.")
    note: str = Field(default="")


class WeeklyReportReading(BaseModel):
    """One historical weather+actual reading (roadmap P2.9). Must cover
    every `forecasts[].target_timestamp` AND at least
    src.evaluation.baselines.LAG_HOURS (24h) before the earliest one --
    persistence/smart_persistence need that lookback."""
    timestamp: str = Field(..., description="ISO8601 timestamp, e.g. '2024-06-01T09:00:00+05:30'.")
    shortwave_radiation: float = Field(..., ge=0, le=1200, description="GHI in W/m^2")
    cloud_cover: float = Field(..., ge=0, le=100)
    temperature_2m: float = Field(..., ge=-10, le=60)
    relative_humidity_2m: float = Field(..., ge=0, le=100)
    wind_speed_10m: float = Field(..., ge=0, le=50)
    solar_output_mw: float = Field(..., description="Actual generation -- the now-known outcome.")


class ServedForecastRecord(BaseModel):
    """One forecast that was actually served, now scoreable against a
    known outcome (roadmap P2.9)."""
    target_timestamp: str = Field(..., description="Which timestamp this forecast was FOR.")
    horizon_hours: int = Field(..., description="How far ahead the forecast was made, e.g. 1, 6, 24.")
    predicted_mw: float


class WeeklyReportRequest(BaseModel):
    """Request body for the weekly forecast-performance report (roadmap
    P2.9)."""
    plant_id: str = Field(default=DEFAULT_PLANT_ID)
    readings: list[WeeklyReportReading] = Field(..., min_length=1)
    forecasts: list[ServedForecastRecord] = Field(..., min_length=1)


class PlantLocationInput(BaseModel):
    """Location block for onboarding a new plant (roadmap P2.3) --
    mirrors configs/plants/<id>.yaml's `location` block."""
    name: str = Field(..., min_length=1, max_length=200)
    state: str = Field(..., min_length=1, max_length=100)
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    elevation_m: float | None = Field(
        default=None, ge=-500, le=9000,
        description="Optional. If omitted, left out of the stored config "
                     "entirely (not written as null) -- every existing "
                     "consumer's `.get('elevation_m', 0.0)` fallback then "
                     "applies exactly as it does today."
    )
    timezone: str = Field(..., description="IANA timezone name, e.g. 'Asia/Kolkata'.")


class PlantCapacityInput(BaseModel):
    ac_capacity_mw: float = Field(..., gt=0, le=10000)
    dc_capacity_mw: float = Field(..., gt=0, le=10000)
    panel_efficiency: float = Field(..., gt=0, le=1)
    temperature_coefficient: float = Field(..., ge=-0.02, le=0)
    performance_ratio: float = Field(
        ..., gt=0, le=1,
        description="Required, never defaulted -- src/attribution/loss.py "
                     "and src/evaluation/baselines.py index this with no "
                     "fallback."
    )
    panel_area_m2: float = Field(..., gt=0)


class PlantGridInput(BaseModel):
    """Only export_limit_mw is captured today -- sldc/rldc/ists_or_instate/
    qca_role/metering_point remain null (zero code reads them)."""
    export_limit_mw: float = Field(
        ..., gt=0, le=10000,
        description="Contractual/regulatory cap on grid export (MW); may "
                     "be less than capacity.ac_capacity_mw. Enforced by "
                     "POST /optimize via a curtailment mechanism."
    )


class PlantEquipmentInput(BaseModel):
    commercial_operation_date: str = Field(..., description="ISO 8601 date (YYYY-MM-DD).")
    module_type: str = Field(..., min_length=1, max_length=200)
    inverter_count: int = Field(..., ge=1, le=100000)


class PlantRegulatoryInput(BaseModel):
    seller_category: str = Field(
        ..., description="Validated against the real WS-seller categories "
                          "src/regulatory/dsm.py implements."
    )
    contract_rate_rs_per_kwh: float | None = Field(default=None, gt=0)
    transaction_type: str | None = Field(
        default=None,
        description="'bilateral' or 'collective' (CERC IEGC 2023 Reg "
                    "49(8)). If omitted, left out of the stored config "
                    "entirely so every existing consumer's documented "
                    "fallback applies unchanged."
    )


class PlantOnboardingRequest(BaseModel):
    """POST /plants request body (roadmap P2.3)."""
    plant_id: str = Field(..., min_length=2, max_length=50)
    name: str = Field(..., min_length=1, max_length=200)
    location: PlantLocationInput
    capacity: PlantCapacityInput
    grid: PlantGridInput
    equipment: PlantEquipmentInput
    regulatory: PlantRegulatoryInput


class PlantOnboardingResponse(BaseModel):
    plant_id: str
    config_path: str
    plant_config: dict


# ── Endpoints ─────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Check if API is running and model is loaded."""
    return {
        "status"      : "healthy",
        "model_loaded": model is not None,
        "timestamp"   : datetime.now().isoformat(),
        "version"     : "1.0.0"
    }


@app.get("/plants")
def plants():
    """List configured plant IDs (each backed by configs/plants/<id>.yaml).
    Pass one of these as `plant_id` in a /forecast request."""
    return {"plant_ids": list_plant_ids()}


@app.get("/plants/{plant_id}")
def plant_detail(plant_id: str):
    """One plant's full configuration (configs/plants/<plant_id>.yaml) --
    exactly what every forecasting/optimization/regulatory endpoint reads
    for this plant_id, so also the way to confirm a freshly onboarded
    plant (POST /plants) is live. Includes data_provenance, which lists
    which values are simulated, placeholders, or assumptions so a client
    can label them rather than present them as facts."""
    # deepcopy: get_plant_config is lru_cached, so the dict is shared.
    return copy.deepcopy(_load_plant(plant_id))


@app.get("/model-card")
def model_card():
    """The served model's card (src/models/model_card.json): training data
    window, feature list, rolling-origin backtest and final-holdout
    metrics against baselines, and quantile-model calibration. All scored
    on synthetic data -- see the README's Honest Results section."""
    if not MODEL_CARD_PATH.exists():
        raise HTTPException(status_code=404, detail="model_card.json not found -- run `make train`.")
    with open(MODEL_CARD_PATH) as f:
        return json.load(f)


@app.get("/forecast/live", response_model=LiveForecastResponse)
def forecast_live(date: str, plant_id: str = DEFAULT_PLANT_ID):
    """
    Forecast one plant-day on the real 96-block/15-minute grid, fetching
    the weather server-side from Open-Meteo for the plant's own location.

    Unlike POST /forecast (caller supplies hourly weather, gets hourly MW),
    this returns 96 mean-MW-per-block values ready to feed straight into
    POST /optimize or POST /schedule/revise. Open-Meteo only serves dates
    from roughly 3 months back to ~16 days ahead; outside that, or if
    Open-Meteo is unreachable, this returns 502 -- it never substitutes
    made-up weather.

    Caveat: the model treats each hourly Open-Meteo row as the weather AT
    that timestamp, matching POST /forecast and the dashboard; Open-Meteo's
    radiation values are actually preceding-hour means.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check server logs.")

    plant_config = _load_plant(plant_id)
    day = _parse_date(date).strftime("%Y-%m-%d")
    capacity_mw = plant_config["capacity"]["ac_capacity_mw"]

    try:
        weather = _fetch_weather_for_date(plant_config, day)
    except Exception as e:
        logger.error(f"Open-Meteo fetch failed for {plant_id} on {day}: {e}")
        raise HTTPException(
            status_code=502,
            detail=f"Weather fetch from Open-Meteo failed for {day}: {e}. "
                    "Open-Meteo's forecast endpoint covers roughly the past 3 months "
                    "to 16 days ahead."
        )

    model_inputs = ["shortwave_radiation", "cloud_cover", "temperature_2m",
                     "relative_humidity_2m", "wind_speed_10m"]
    missing = [c for c in model_inputs if c not in weather.columns]
    if missing:
        raise HTTPException(status_code=502, detail=f"Open-Meteo response missing {missing}")
    if weather[model_inputs].isnull().any().any():
        raise HTTPException(
            status_code=502,
            detail=f"Open-Meteo returned null values for {day} -- refusing to "
                    "forecast from gap-filled weather."
        )

    try:
        raw = weather[model_inputs].astype(float).copy()
        raw["hour"] = weather.index.hour
        raw["month"] = weather.index.month
        point, q_preds = _predict_mw(raw, plant_config, capacity_mw)

        block_starts = block_boundaries(day)[:-1]
        point_blocks = _hourly_mw_to_blocks(point, weather.index, day)
        p10 = p50 = p90 = []
        if q_preds is not None:
            p10, p50, p90 = (_hourly_mw_to_blocks(q_preds[:, i], weather.index, day) for i in range(3))

        peak_idx = int(np.argmax(point_blocks))
        hourly = weather.index < block_boundaries(day)[-1]  # drop the trailing next-day 00:00 row
        return LiveForecastResponse(
            plant_id             = plant_id,
            date                 = day,
            capacity_mw          = capacity_mw,
            block_starts         = [ts.isoformat() for ts in block_starts],
            predictions_mw       = point_blocks,
            predictions_p10_mw   = p10,
            predictions_p50_mw   = p50,
            predictions_p90_mw   = p90,
            peak_output_mw       = point_blocks[peak_idx],
            peak_block_start     = block_starts[peak_idx].isoformat(),
            total_generation_mwh = round(sum(point_blocks) * BLOCK_MINUTES / 60, 2),
            weather_hourly       = {
                "timestamps": [ts.isoformat() for ts in weather.index[hourly]],
                **{c: [round(float(v), 2) for v in weather.loc[hourly, c]] for c in model_inputs},
            },
            weather_source       = "open-meteo",
            generated_at         = datetime.now().isoformat(),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Live forecast error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/dsm/estimate")
def dsm_estimate(request: DsmEstimateRequest):
    """
    Estimate CERC DSM deviation charges for a schedule vs an injection
    profile (Regulation 8(4) tiered WS-seller structure,
    src/regulatory/dsm.py), per block and in total.

    Use it for "what would this forecast/dispatch cost against my
    declared schedule" -- e.g. P50 forecast vs schedule, or /optimize's
    grid delivery vs schedule. net_rs is cost-positive: > 0 is payable by
    the seller, < 0 receivable. Rupee figures are only as real as the
    plant's contract_rate_rs_per_kwh -- see GET /plants/{plant_id}'s
    data_provenance.
    """
    plant_config = _load_plant(request.plant_id)
    regulatory = plant_config.get("regulatory") or {}
    contract_rate = regulatory.get("contract_rate_rs_per_kwh")
    if contract_rate is None:
        raise HTTPException(
            status_code=422,
            detail=f"plant_id={request.plant_id!r} has no regulatory.contract_rate_rs_per_kwh configured."
        )
    if len(request.scheduled_mw) != len(request.injected_mw):
        raise HTTPException(
            status_code=422,
            detail=f"scheduled_mw and injected_mw must be the same length, got "
                   f"{len(request.scheduled_mw)} and {len(request.injected_mw)}."
        )

    dt_hours = request.dt_hours
    block_starts = None
    as_of = datetime.now().date()
    if request.date is not None:
        as_of = _parse_date(request.date).date()
        if len(request.scheduled_mw) != BLOCKS_PER_DAY:
            raise HTTPException(
                status_code=422,
                detail=f"date={request.date!r} given: both lists must have exactly "
                       f"{BLOCKS_PER_DAY} entries, got {len(request.scheduled_mw)}."
            )
        dt_hours = BLOCK_MINUTES / 60
        block_starts = block_boundaries(request.date)[:-1]

    capacity_mw = plant_config["capacity"]["ac_capacity_mw"]
    available_capacity_mwh = capacity_mw * dt_hours
    category = regulatory.get("seller_category", "solar")

    blocks = []
    for i, (sched, inj) in enumerate(zip(request.scheduled_mw, request.injected_mw)):
        deviation_mwh = (inj - sched) * dt_hours
        settlement = dsm.deviation_settlement(
            deviation_mwh, available_capacity_mwh, contract_rate, category, as_of,
        )
        row = {
            "block": i + 1,
            "scheduled_mw": sched,
            "injected_mw": inj,
            "deviation_mwh": round(deviation_mwh, 4),
            "deviation_pct": round(dsm.deviation_pct(inj * dt_hours, sched * dt_hours, available_capacity_mwh), 3),
            "net_rs": round(settlement["net_rs"], 2),
        }
        if block_starts is not None:
            row["block_start"] = block_starts[i].isoformat()
        blocks.append(row)

    net = [b["net_rs"] for b in blocks]
    provenance = plant_config.get("data_provenance") or {}
    return {
        "plant_id": request.plant_id,
        "dsm_ruleset_id": regulatory.get("dsm_ruleset_id"),
        "seller_category": category,
        "contract_rate_rs_per_kwh": contract_rate,
        "contract_rate_is_placeholder": "regulatory.contract_rate_rs_per_kwh"
                                         in (provenance.get("placeholder_fields") or []),
        "as_of": as_of.isoformat(),
        "dt_hours": dt_hours,
        "blocks": blocks,
        "summary": {
            "total_net_rs": round(sum(net), 2),
            "total_payable_rs": round(sum(v for v in net if v > 0), 2),
            "total_receivable_rs": round(-sum(v for v in net if v < 0), 2),
            "total_abs_deviation_mwh": round(sum(abs(b["deviation_mwh"]) for b in blocks), 4),
        },
    }


@app.post("/forecast", response_model=ForecastResponse)
def forecast(request: ForecastRequest):
    """
    Generate solar output forecast from weather inputs, for one configured
    plant (see GET /plants).

    Takes weather data at real timestamps (roadmap P1.4: no more anchoring
    to a fake placeholder date) and returns predicted solar generation for
    each one.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check server logs."
        )

    try:
        plant_config = get_plant_config(request.plant_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    capacity_mw = request.plant_capacity_mw or plant_config["capacity"]["ac_capacity_mw"]

    try:
        # Build the feature DataFrame via the single canonical pipeline
        # (src/features/pipeline.py) instead of a hand-built, easily
        # drifting dict. build_features needs a real timestamp (to compute
        # a solar-position clear-sky estimate, using THIS plant's lat/lon) --
        # each WeatherInput now carries its own genuine ISO timestamp
        # (roadmap P1.4), replacing the earlier "anchor every request to a
        # fixed reference year/day" placeholder this endpoint used before
        # P1.2's time-block infrastructure existed. hour/month, needed by
        # build_features' cyclical encodings, are derived from the real
        # timestamp rather than supplied separately (so they can never
        # disagree with it).
        timestamps = build_timestamp_index([h.timestamp for h in request.hours])
        raw = pd.DataFrame(
            [{
                "hour": ts.hour,
                "month": ts.month,
                "cloud_cover": h.cloud_cover,
                "shortwave_radiation": h.shortwave_radiation,
                "temperature_2m": h.temperature_2m,
                "relative_humidity_2m": h.relative_humidity_2m,
                "wind_speed_10m": h.wind_speed_10m,
            } for h, ts in zip(request.hours, timestamps)],
            index=timestamps,
        )
        point, q_preds = _predict_mw(raw, plant_config, capacity_mw)
        predictions = point.tolist()

        peak_idx = int(np.argmax(predictions))

        # Energy = power x duration. Summing MW only equals MWh for hourly
        # inputs -- a 96-block (15-min) request used to report 4x the true
        # energy. Use the median spacing; a single timestamp is 1 hour.
        if len(timestamps) > 1:
            interval_hours = float(np.median(timestamps.to_series().diff().dropna().dt.total_seconds()) / 3600)
        else:
            interval_hours = 1.0

        logger.info(
            f"Forecast generated: {len(predictions)} hours, "
            f"peak {max(predictions):.1f} MW at {timestamps[peak_idx].isoformat()}"
        )

        p10_mw, p50_mw, p90_mw = [], [], []
        if q_preds is not None:
            p10_mw = [round(v, 2) for v in q_preds[:, 0].tolist()]
            p50_mw = [round(v, 2) for v in q_preds[:, 1].tolist()]
            p90_mw = [round(v, 2) for v in q_preds[:, 2].tolist()]

        return ForecastResponse(
            forecast_hours      = len(predictions),
            timestamps          = [ts.isoformat() for ts in timestamps],
            predictions_mw      = [round(p, 2) for p in predictions],
            peak_output_mw      = round(max(predictions), 2),
            peak_index          = peak_idx,
            peak_timestamp      = timestamps[peak_idx].isoformat(),
            total_generation_mwh= round(sum(predictions) * interval_hours, 2),
            interval_hours      = interval_hours,
            generated_at        = datetime.now().isoformat(),
            predictions_p10_mw  = p10_mw,
            predictions_p50_mw  = p50_mw,
            predictions_p90_mw  = p90_mw,
        )

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Forecast error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/optimize")
def optimize(request: OptimizeRequest):
    """
    Run battery dispatch optimization.

    Takes a solar forecast and the plant's declared delivery schedule,
    returns a charge/discharge schedule minimizing deviation from that
    schedule (DSM exposure), not "unmet demand" -- a solar IPP has a
    schedule, not demand (roadmap P1.6).

    If `date` is given, both input lists must align to the real 96-block/
    15-minute grid (roadmap P1.4, src/time_blocks.py) -- exactly 96
    entries, and dt_hours is fixed at 0.25 regardless of what was passed.
    """
    block_starts = None
    if request.date is not None:
        if len(request.solar_forecast_mw) != BLOCKS_PER_DAY or len(request.declared_schedule_mw) != BLOCKS_PER_DAY:
            raise HTTPException(
                status_code=422,
                detail=f"date={request.date!r} given: solar_forecast_mw and declared_schedule_mw "
                       f"must each have exactly {BLOCKS_PER_DAY} entries (one per real 15-minute "
                       f"block of that date), got {len(request.solar_forecast_mw)} and "
                       f"{len(request.declared_schedule_mw)}."
            )
        block_starts = block_boundaries(request.date)[:-1]  # 96 block-start timestamps

    dsm_kwargs = {}
    export_limit_mw = None
    if request.plant_id is not None:
        try:
            plant_config = get_plant_config(request.plant_id)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        regulatory = plant_config.get("regulatory") or {}
        contract_rate = regulatory.get("contract_rate_rs_per_kwh")
        if contract_rate is not None:
            dsm_kwargs = dict(
                dsm_contract_rate_rs_per_kwh=contract_rate,
                dsm_available_capacity_mw=plant_config["capacity"]["ac_capacity_mw"],
                dsm_seller_category=regulatory.get("seller_category", "solar"),
                # The schedule's own date picks the pre/post-01.04.2026
                # volume limits; without one, the optimizer uses today.
                **({"dsm_as_of": _parse_date(request.date).date()} if request.date else {}),
            )
        else:
            logger.warning(
                f"plant_id={request.plant_id!r} has no regulatory.contract_rate_rs_per_kwh "
                f"configured -- falling back to the flat deviation_penalty_per_mwh."
            )
        # Grid export cap (roadmap P2.3) -- a PHYSICAL constraint on the
        # interconnection, independent of whether DSM mode is active
        # above (a commercial concern), so it's read and applied
        # regardless of dsm_kwargs.
        export_limit_mw = (plant_config.get("grid") or {}).get("export_limit_mw")

    try:
        optimizer = BatteryOptimizer(
            battery_capacity_mwh  = request.battery_capacity_mwh,
            charge_rate_mw        = request.charge_rate_mw,
            discharge_rate_mw     = request.discharge_rate_mw,
            initial_charge_mwh    = request.initial_charge_mwh,
            dt_hours              = 0.25 if block_starts is not None else request.dt_hours,
            round_trip_efficiency = request.round_trip_efficiency,
            export_limit_mw       = export_limit_mw,
            **dsm_kwargs,
        )

        results = optimizer.optimize(
            solar_forecast        = np.array(request.solar_forecast_mw),
            declared_schedule_mw  = np.array(request.declared_schedule_mw),
        )

        schedule_records = []
        for i, record in enumerate(results.to_dict(orient="records")):
            clean = {k: float(v) if hasattr(v, 'item') else v
                     for k, v in record.items()}
            if block_starts is not None:
                clean["block_start"] = block_starts[i].isoformat()
            schedule_records.append(clean)

        return {
            "status": "optimal",
            "blocks": int(len(results)),
            "schedule": schedule_records,
            "summary": {
                # charge_mw/discharge_mw are RATES (MW); multiply by the
                # block duration to get energy (MWh) -- summing rates
                # directly only happened to work before P1.5 because Δt
                # was implicitly always 1 hour.
                "total_charged_mwh": float(round(results['charge_mw'].sum() * optimizer.dt_hours, 2)),
                "total_discharged_mwh": float(round(results['discharge_mw'].sum() * optimizer.dt_hours, 2)),
                "total_deviation_mwh": float(round(results['deviation_mwh'].sum(), 2)),
                **({"total_dsm_rs": float(round(results['dsm_rs'].sum(), 2))}
                   if optimizer.dsm_enabled else {}),
                "blocks_charging": int((results['action'] == 'CHARGE').sum()),
                "blocks_discharging": int((results['action'] == 'DISCHARGE').sum()),
                "blocks_hold": int((results['action'] == 'HOLD').sum()),
            }
        }
    except Exception as e:
        logger.error(f"Optimization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/schedule/revise", response_model=ReviseScheduleResponse)
def revise_schedule(request: ReviseScheduleRequest):
    """
    Apply a proposed schedule revision under the REAL CERC IEGC 2023 gate-
    closure timing (roadmap P1.4) -- see src/regulatory/grid_code.py.

    A day-ahead schedule cannot simply be overwritten the moment a better
    forecast arrives: Regulation 49(4)(c) delays a requested revision by
    6-7 more full time blocks beyond the block the request itself falls
    in, and Regulation 49(8) only permits a WS seller to revise at all if
    it sells under a bilateral transaction structure (this plant's
    configured regulatory.revision_windows.transaction_type). Blocks
    before the computed effective timestamp are returned UNCHANGED from
    locked_schedule_mw regardless of what proposed_schedule_mw says.
    """
    try:
        plant_config = get_plant_config(request.plant_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    revision_windows = (plant_config.get("regulatory") or {}).get("revision_windows") or {}
    transaction_type = revision_windows.get("transaction_type", "bilateral")

    try:
        edges = block_boundaries(request.date)[:-1]  # 96 block-start timestamps
        locked = pd.Series(request.locked_schedule_mw, index=edges)
        proposed = pd.Series(request.proposed_schedule_mw, index=edges)

        result = apply_schedule_revision(
            locked, proposed, request.request_timestamp, transaction_type=transaction_type,
        )

        return ReviseScheduleResponse(
            applied_schedule_mw = [round(v, 2) for v in result["applied_schedule"].tolist()],
            effective_timestamp = (
                result["effective_timestamp"].isoformat() if result["effective_timestamp"] is not None else None
            ),
            revision_allowed    = result["revision_allowed"],
            locked_block_count  = result["locked_block_count"],
        )
    except Exception as e:
        logger.error(f"Schedule revision error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/schedule/gate-closures")
def gate_closures(timestamp: str, plant_id: str = DEFAULT_PLANT_ID):
    """
    The next real gate-closure instants an operator would actually face at
    `timestamp` "now" (roadmap P1.4) -- see src/regulatory/grid_code.py.

    Two independent mechanisms, both from CERC IEGC 2023 Regulation 49:
      - `bilateral_revision`: when a Regulation 49(8) schedule revision
        requested right now would take effect (Regulation 49(4)(c)) --
        only usable at all if this plant's configured transaction_type
        is "bilateral" (see GET /plants and configs/plants/*.yaml).
      - `real_time_market`: the current half-hour RTM delivery window and
        its bid-window-open / gate-closure instants (Regulation 49(1)(q)).
    """
    try:
        plant_config = get_plant_config(plant_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    revision_windows = (plant_config.get("regulatory") or {}).get("revision_windows") or {}
    transaction_type = revision_windows.get("transaction_type", "bilateral")

    try:
        now = pd.Timestamp(timestamp)
        bilateral_allowed = grid_code.can_revise_schedule(transaction_type)

        return {
            "timestamp": now.isoformat(),
            "plant_id": plant_id,
            "transaction_type": transaction_type,
            "bilateral_revision": {
                "allowed": bilateral_allowed,
                "effective_timestamp": (
                    grid_code.revision_effective_timestamp(now).isoformat() if bilateral_allowed else None
                ),
            },
            "real_time_market": {
                "delivery_window_start": grid_code.rtm_delivery_window_start(now).isoformat(),
                "bid_window_open": grid_code.rtm_bid_window_open_timestamp(now).isoformat(),
                "gate_closure": grid_code.rtm_gate_closure_timestamp(now).isoformat(),
            },
        }
    except Exception as e:
        logger.error(f"Gate closure lookup error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/quality/check")
def quality_check(request: QualityCheckRequest):
    """
    Run the data-quality gate over a plant's raw generation readings
    (roadmap P2.2) -- "the customer sees a quality report before they see
    a forecast." Detects timestamp gaps, duplicates, a missing timezone,
    negative power, values above the plant's AC capacity, flatlines
    (stuck inverter), and night-time non-zero generation (meter/timezone
    fault). See src/quality/gate.py for the exact rules and thresholds.
    """
    try:
        plant_config = get_plant_config(request.plant_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    try:
        # A plain pd.Index, not build_timestamp_index -- this endpoint's
        # entire job is diagnosing raw data problems, including a mix of
        # tz-aware and tz-naive timestamps, which falls back to a generic
        # object-dtype Index here (see src/quality/gate.py's
        # _check_missing_timezone) instead of being rejected outright.
        timestamps = pd.Index([pd.Timestamp(r.timestamp) for r in request.readings])
        df = pd.DataFrame(
            {"solar_output_mw": [r.power_mw for r in request.readings]},
            index=timestamps,
        )
        report = run_quality_checks(df, plant_config)
        return report.to_dict()
    except Exception as e:
        logger.error(f"Quality check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/losses/attribute")
def losses_attribute(request: LossAttributionRequest):
    """
    Classify every block with a material generation shortfall against
    expected-from-irradiance into a cause -- weather, equipment
    (suspected), curtailment (possible), or unknown -- with the evidence
    that produced each call (roadmap P2.4). When at least 10 distinct
    calendar days of readings are supplied, also runs the separate
    day-level soiling check (a slow, sustained decline in the daily
    actual/expected ratio -- not visible at single-block granularity).

    See src/attribution/loss.py's module docstring for exactly what
    "equipment" and "curtailment" here can and cannot confirm: this
    platform has no per-inverter telemetry and no real grid
    curtailment-instruction feed, so those two causes are always labelled
    suspected/possible, never a confirmed asset or a confirmed grid order.
    """
    try:
        plant_config = get_plant_config(request.plant_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    try:
        timestamps = build_timestamp_index([r.timestamp for r in request.readings])
        df = pd.DataFrame(
            {
                "solar_output_mw": [r.power_mw for r in request.readings],
                "shortwave_radiation": [r.ghi_w_m2 for r in request.readings],
                "temperature_2m": [r.temperature_c for r in request.readings],
            },
            index=timestamps,
        )
        report = attribute_losses(df, plant_config)
        result = report.to_dict()

        daily_ratio = daily_performance_ratio(df, plant_config)
        soiling = detect_soiling(daily_ratio)
        result["soiling"] = soiling.to_dict() if soiling is not None else None
        result["days_observed_for_soiling_check"] = len(daily_ratio)

        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Loss attribution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/schedule/risk")
def schedule_risk(request: ScheduleRiskRequest):
    """
    Score each block's DSM exposure risk (roadmap P2.5) by running its
    P10/P50/P90 forecast against the declared schedule through the real
    CERC DSM settlement math (src/regulatory/dsm.py, roadmap P1.7) and
    reporting which Note-1 volume-limit band each quantile reaches. See
    src/risk/schedule_risk.py for exactly how risk levels are derived.

    DISCLAIMER (present on every block and on the report itself): this is
    an indicative estimate from configured assumptions and uploaded data,
    not an official settlement statement.
    """
    lists = [request.declared_schedule_mw, request.p10_mw, request.p50_mw, request.p90_mw]

    block_starts = None
    if request.date is not None:
        if any(len(l) != BLOCKS_PER_DAY for l in lists):
            raise HTTPException(
                status_code=422,
                detail=f"date={request.date!r} given: declared_schedule_mw, p10_mw, p50_mw "
                       f"and p90_mw must each have exactly {BLOCKS_PER_DAY} entries (one per "
                       f"real 15-minute block of that date), got {[len(l) for l in lists]}."
            )
        block_starts = block_boundaries(request.date)[:-1]
    elif len({len(l) for l in lists}) != 1:
        raise HTTPException(
            status_code=422,
            detail="declared_schedule_mw, p10_mw, p50_mw and p90_mw must all have the same "
                   f"length, got {[len(l) for l in lists]}."
        )

    try:
        plant_config = get_plant_config(request.plant_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    as_of_str = request.as_of or request.date
    as_of = pd.Timestamp(as_of_str).date() if as_of_str else datetime.now().date()
    dt_hours = 0.25 if block_starts is not None else request.dt_hours

    try:
        report = score_schedule_risk(
            declared_schedule_mw=request.declared_schedule_mw,
            p10_mw=request.p10_mw,
            p50_mw=request.p50_mw,
            p90_mw=request.p90_mw,
            dt_hours=dt_hours,
            plant_config=plant_config,
            as_of=as_of,
            timestamps=block_starts,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return report.to_dict()


@app.post("/recommendations/generate")
def recommendations_generate(request: RecommendationsGenerateRequest):
    """
    Generate operator recommendations (roadmap P2.6) -- schedule
    revision and battery action from a schedule-risk report (needs
    `schedule_risk`), inspection from a loss-attribution report (needs
    `loss_readings`). Every recommendation is logged to the append-only
    recommendation log (src/recommendations/store.py) as `pending` and
    returned; NOTHING here is applied automatically -- an operator must
    separately call POST /recommendations/{id}/decide.
    """
    try:
        plant_config = get_plant_config(request.plant_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    generated = []

    if request.schedule_risk is not None:
        sr = request.schedule_risk
        lists = [sr.declared_schedule_mw, sr.p10_mw, sr.p50_mw, sr.p90_mw]

        block_starts = None
        if sr.date is not None:
            if any(len(l) != BLOCKS_PER_DAY for l in lists):
                raise HTTPException(
                    status_code=422,
                    detail=f"schedule_risk.date={sr.date!r} given: declared_schedule_mw, p10_mw, "
                           f"p50_mw and p90_mw must each have exactly {BLOCKS_PER_DAY} entries, "
                           f"got {[len(l) for l in lists]}."
                )
            block_starts = block_boundaries(sr.date)[:-1]
        elif len({len(l) for l in lists}) != 1:
            raise HTTPException(
                status_code=422,
                detail="schedule_risk's declared_schedule_mw, p10_mw, p50_mw and p90_mw must "
                       f"all have the same length, got {[len(l) for l in lists]}."
            )

        as_of_str = sr.as_of or sr.date
        as_of = pd.Timestamp(as_of_str).date() if as_of_str else datetime.now().date()
        dt_hours = 0.25 if block_starts is not None else sr.dt_hours

        try:
            risk_report = score_schedule_risk(
                declared_schedule_mw=sr.declared_schedule_mw,
                p10_mw=sr.p10_mw, p50_mw=sr.p50_mw, p90_mw=sr.p90_mw,
                dt_hours=dt_hours, plant_config=plant_config, as_of=as_of,
                timestamps=block_starts,
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

        generated += recommend_schedule_revisions(risk_report, plant_config, now=request.now)
        battery_state = request.battery_state.model_dump() if request.battery_state is not None else None
        generated += recommend_battery_actions(risk_report, battery_state, dt_hours)

    if request.loss_readings is not None:
        try:
            timestamps = build_timestamp_index([r.timestamp for r in request.loss_readings])
            loss_df = pd.DataFrame(
                {
                    "solar_output_mw": [r.power_mw for r in request.loss_readings],
                    "shortwave_radiation": [r.ghi_w_m2 for r in request.loss_readings],
                    "temperature_2m": [r.temperature_c for r in request.loss_readings],
                },
                index=timestamps,
            )
            loss_report = attribute_losses(loss_df, plant_config)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except Exception as e:
            logger.error(f"Recommendation loss-attribution error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

        generated += recommend_inspections(loss_report)

    conn = get_recommendations_db()
    logged = []
    for rec in generated:
        rec_id = recommendations_store.log_recommendation(
            conn,
            plant_id=request.plant_id,
            recommendation_type=rec["recommendation_type"],
            trigger=rec["trigger"],
            evidence=rec["evidence"],
            suggested_action=rec["suggested_action"],
            block_timestamp=rec["timestamp"],
        )
        logged.append(recommendations_store.get_recommendation(conn, rec_id))

    return {"generated": len(logged), "recommendations": logged}


@app.get("/recommendations")
def recommendations_list(plant_id: str | None = None, status: str | None = None):
    """List logged recommendations (roadmap P2.6), newest first. Filter
    with ?plant_id=... and/or ?status=pending|approved|dismissed."""
    conn = get_recommendations_db()
    return {"recommendations": recommendations_store.list_recommendations(conn, plant_id=plant_id, status=status)}


@app.post("/recommendations/{recommendation_id}/decide")
def recommendations_decide(recommendation_id: int, request: RecommendationDecisionRequest):
    """
    Record a human operator's explicit approve/dismiss decision on a
    recommendation (roadmap P2.6). This only logs the decision -- it
    never itself revises a schedule, moves a battery, or does anything
    else; that action, if taken, happens outside this platform, by the
    operator, exactly as the roadmap requires ("never automatic
    control").
    """
    conn = get_recommendations_db()
    try:
        return recommendations_store.decide_recommendation(
            conn, recommendation_id, request.decision, request.decided_by, request.note
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/reports/weekly")
def reports_weekly(request: WeeklyReportRequest):
    """
    Score a batch of already-served forecasts against their now-known
    actual outcomes (roadmap P2.9), broken down by lead time (horizon)
    and time of day (block), against the same three untrained baselines
    used in this project's own training-time evaluation
    (src/evaluation/baselines.py, shared with benchmark.py so both use
    identical math). Every number here is traced directly to the
    `readings`/`forecasts` given -- nothing is estimated or assumed.
    """
    try:
        plant_config = get_plant_config(request.plant_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    try:
        timestamps = build_timestamp_index([r.timestamp for r in request.readings])
        raw = pd.DataFrame(
            [{
                "hour": ts.hour,
                "month": ts.month,
                "cloud_cover": r.cloud_cover,
                "shortwave_radiation": r.shortwave_radiation,
                "temperature_2m": r.temperature_2m,
                "relative_humidity_2m": r.relative_humidity_2m,
                "wind_speed_10m": r.wind_speed_10m,
                "solar_output_mw": r.solar_output_mw,
            } for r, ts in zip(request.readings, timestamps)],
            index=timestamps,
        )
        features = build_features(raw, plant_config)

        forecasts_df = pd.DataFrame([{
            "target_timestamp": f.target_timestamp,
            "horizon_hours": f.horizon_hours,
            "predicted_mw": f.predicted_mw,
        } for f in request.forecasts])

        report = generate_weekly_report(forecasts_df, features, plant_config)
        return report.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Weekly report error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/plants", response_model=PlantOnboardingResponse, status_code=201)
def onboard_plant(request: PlantOnboardingRequest):
    """
    Self-serve plant onboarding (roadmap P2.3): "self-serve capture of
    everything in P1.1 plus COD, module type, inverter count, grid export
    limit." Acceptance criterion: "a new plant is live in under 30
    minutes without your involvement" -- this endpoint writes
    configs/plants/<plant_id>.yaml directly, the exact file every other
    endpoint already reads via get_plant_config(), so a freshly onboarded
    plant is immediately usable by GET /plants and every other endpoint,
    with no restart and no code change.

    SCOPED TO CONFIGURATION, NOT DATA: does not depend on roadmap P2.1
    (CSV/Excel historical data upload, not yet built -- blocked on real
    customer files). This only captures the plant metadata every
    endpoint's plant_config argument needs; a freshly onboarded plant has
    no historical generation data of its own until P2.1 exists or a
    caller supplies readings directly (e.g. to POST /quality/check).

    CREATE-ONLY: 409 if plant_id is already registered. There is no
    update/edit endpoint -- get_plant_config() is @lru_cache'd, and
    safely invalidating that cache for an in-place edit is future scope
    (see README Known Gaps).

    grid.export_limit_mw IS enforced, not just stored: POST /optimize
    reads it and caps dispatch via a curtailment mechanism in
    src/optimization/battery_optimizer.py.
    """
    try:
        config = register_plant(
            plant_id=request.plant_id,
            name=request.name,
            location=request.location.model_dump(),
            capacity=request.capacity.model_dump(),
            grid=request.grid.model_dump(),
            equipment=request.equipment.model_dump(),
            regulatory=request.regulatory.model_dump(),
        )
    except PlantAlreadyRegisteredError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Plant onboarding failed for {request.plant_id!r}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return PlantOnboardingResponse(
        plant_id=config["plant_id"],
        config_path=f"configs/plants/{config['plant_id']}.yaml",
        plant_config=config,
    )
