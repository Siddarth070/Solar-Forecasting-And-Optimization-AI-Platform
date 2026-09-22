import numpy as np
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from loguru import logger
from xgboost import XGBRegressor

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.pipeline import build_features, SERVING_FEATURE_COLUMNS
from src.optimization.battery_optimizer import BatteryOptimizer
from src.scheduling.rolling_horizon import apply_schedule_revision
from src.time_blocks import BLOCKS_PER_DAY, block_boundaries
from src.utils.config_loader import get_plant_config, list_plant_ids

DEFAULT_PLANT_ID = "jaipur_100mw"

# ── App setup ─────────────────────────────────────────────────
app = FastAPI(
    title="Solar Forecast Platform",
    description="AI-based solar energy forecasting and grid optimization, "
                 "for any plant configured under configs/plants/",
    version="1.0.0"
)

# ── Load model on startup ─────────────────────────────────────
# JSON, not pickle: a pickle can silently break across library versions
# and is opaque to review (roadmap P0.6).
MODEL_PATH = PROJECT_ROOT / "src" / "models" / "xgboost_solar_v2.json"
QUANTILE_MODEL_PATH = PROJECT_ROOT / "src" / "models" / "xgboost_solar_quantile.json"

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
    """Input weather data for one hour."""
    shortwave_radiation   : float = Field(..., ge=0, le=1200, description="GHI in W/m²")
    cloud_cover           : float = Field(..., ge=0, le=100,  description="Cloud cover %")
    temperature_2m        : float = Field(..., ge=-10, le=60, description="Temperature °C")
    relative_humidity_2m  : float = Field(..., ge=0, le=100,  description="Humidity %")
    wind_speed_10m        : float = Field(..., ge=0, le=50,   description="Wind speed m/s")
    hour                  : int   = Field(..., ge=0, le=23,   description="Hour of day")
    month                 : int   = Field(..., ge=1, le=12,   description="Month")


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
    predictions_mw     : list[float]
    peak_output_mw     : float
    peak_hour          : int
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


@app.post("/forecast", response_model=ForecastResponse)
def forecast(request: ForecastRequest):
    """
    Generate solar output forecast from weather inputs, for one configured
    plant (see GET /plants).

    Takes hourly weather data and returns predicted
    solar generation for each hour.
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
        # a solar-position clear-sky estimate, using THIS plant's lat/lon);
        # WeatherInput only carries hour + month, not a full date, so we
        # anchor every request to a fixed reference year/day. This is an
        # approximation pending P1.2 (a proper per-plant, per-timestamp
        # request schema) — it does not affect clear-sky GHI meaningfully
        # within a given hour/month.
        timestamps = pd.DatetimeIndex([
            pd.Timestamp(year=2024, month=h.month, day=15, hour=h.hour,
                         tz="Asia/Kolkata")
            for h in request.hours
        ])
        raw = pd.DataFrame(
            [{
                "hour": h.hour,
                "month": h.month,
                "cloud_cover": h.cloud_cover,
                "shortwave_radiation": h.shortwave_radiation,
                "temperature_2m": h.temperature_2m,
                "relative_humidity_2m": h.relative_humidity_2m,
                "wind_speed_10m": h.wind_speed_10m,
            } for h in request.hours],
            index=timestamps,
        )
        features = build_features(raw, plant_config)
        X = features[SERVING_FEATURE_COLUMNS]
        # Model outputs capacity FRACTION (0-1), not MW (see
        # src/models/train.py) -- rescale by the requested plant's own
        # capacity so one model correctly serves differently-sized plants.
        predictions = model.predict(X) * capacity_mw
        predictions = np.clip(
            predictions, 0, capacity_mw
        ).tolist()

        peak_idx = int(np.argmax(predictions))

        logger.info(
            f"Forecast generated: {len(predictions)} hours, "
            f"peak {max(predictions):.1f} MW at hour {peak_idx}"
        )

        p10_mw, p50_mw, p90_mw = [], [], []
        if quantile_model is not None:
            # Quantile model outputs capacity FRACTION, three columns in
            # [P10, P50, P90] order -- rescale, then sort defensively so a
            # rare crossing never produces P10 > P90 (see benchmark.py).
            q_preds = np.clip(quantile_model.predict(X) * capacity_mw, 0, capacity_mw)
            q_preds = np.sort(q_preds, axis=1)
            p10_mw = [round(v, 2) for v in q_preds[:, 0].tolist()]
            p50_mw = [round(v, 2) for v in q_preds[:, 1].tolist()]
            p90_mw = [round(v, 2) for v in q_preds[:, 2].tolist()]

        return ForecastResponse(
            forecast_hours      = len(predictions),
            predictions_mw      = [round(p, 2) for p in predictions],
            peak_output_mw      = round(max(predictions), 2),
            peak_hour           = peak_idx,
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
    """
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
            dt_hours              = request.dt_hours,
            round_trip_efficiency = request.round_trip_efficiency,
            **dsm_kwargs,
        )

        results = optimizer.optimize(
            solar_forecast        = np.array(request.solar_forecast_mw),
            declared_schedule_mw  = np.array(request.declared_schedule_mw),
        )

        schedule_records = []
        for record in results.to_dict(orient="records"):
            clean = {k: float(v) if hasattr(v, 'item') else v
                     for k, v in record.items()}
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