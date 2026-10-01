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

from src.features.pipeline import build_features, SERVING_FEATURE_COLUMNS
from src.ingestion.open_meteo_fetcher import OpenMeteoFetcher
from src.optimization.battery_optimizer import BatteryOptimizer
from src.regulatory import dsm, grid_code
from src.scheduling.rolling_horizon import apply_schedule_revision
from src.time_blocks import BLOCKS_PER_DAY, BLOCK_MINUTES, block_boundaries, integrate_to_blocks
from src.utils.config_loader import get_plant_config, list_plant_ids

DEFAULT_PLANT_ID = "jaipur_100mw"

# ── App setup ─────────────────────────────────────────────────
app = FastAPI(
    title="Solar Forecast Platform",
    description="AI-based solar energy forecasting and grid optimization, "
                 "for any plant configured under configs/plants/",
    version="1.0.0"
)

# ── CORS (browser frontend, frontend/) ────────────────────────
# Explicit allow-list, never "*": set CORS_ALLOW_ORIGINS to a comma-
# separated list of origins in each deployment. Defaults cover the
# frontend's local dev servers only.
CORS_ALLOW_ORIGINS = [
    o.strip() for o in os.environ.get(
        "CORS_ALLOW_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",") if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
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
    total_generation_mwh: float
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
    """One plant's full configuration (configs/plants/<plant_id>.yaml):
    location, capacity, grid, regulatory, and data_provenance -- the
    latter lists which values are simulated, placeholders, or assumptions
    so a client can label them rather than present them as facts."""
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
        timestamps = pd.DatetimeIndex([pd.Timestamp(h.timestamp) for h in request.hours])
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
            total_generation_mwh= round(sum(predictions), 2),
            generated_at        = datetime.now().isoformat(),
            predictions_p10_mw  = p10_mw,
            predictions_p50_mw  = p50_mw,
            predictions_p90_mw  = p90_mw,
        )

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

    try:
        optimizer = BatteryOptimizer(
            battery_capacity_mwh  = request.battery_capacity_mwh,
            charge_rate_mw        = request.charge_rate_mw,
            discharge_rate_mw     = request.discharge_rate_mw,
            initial_charge_mwh    = request.initial_charge_mwh,
            dt_hours              = 0.25 if block_starts is not None else request.dt_hours,
            round_trip_efficiency = request.round_trip_efficiency,
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