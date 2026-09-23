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

from src.attribution.loss import attribute_losses, daily_performance_ratio, detect_soiling
from src.features.pipeline import build_features, SERVING_FEATURE_COLUMNS
from src.optimization.battery_optimizer import BatteryOptimizer
from src.quality.gate import run_quality_checks
from src.recommendations import store as recommendations_store
from src.recommendations.engine import (
    recommend_battery_actions,
    recommend_inspections,
    recommend_schedule_revisions,
)
from src.regulatory import grid_code
from src.risk.schedule_risk import score_schedule_risk
from src.scheduling.rolling_horizon import apply_schedule_revision
from src.time_blocks import BLOCKS_PER_DAY, block_boundaries
from src.utils.config_loader import get_plant_config, list_plant_ids

DEFAULT_PLANT_ID = "jaipur_100mw"


def get_recommendations_db():
    """A fresh connection to the recommendation log (roadmap P2.6) on
    every call -- SQLite handles this cheaply, and it keeps request
    handling stateless. Tests override this (see
    tests/test_recommendations_api.py) to point at one shared in-memory
    connection instead of the real on-disk log."""
    return recommendations_store.connect()


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
            f"peak {max(predictions):.1f} MW at {timestamps[peak_idx].isoformat()}"
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
        timestamps = pd.DatetimeIndex([pd.Timestamp(r.timestamp) for r in request.readings])
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
        timestamps = pd.DatetimeIndex([pd.Timestamp(r.timestamp) for r in request.readings])
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
            timestamps = pd.DatetimeIndex([pd.Timestamp(r.timestamp) for r in request.loss_readings])
            loss_df = pd.DataFrame(
                {
                    "solar_output_mw": [r.power_mw for r in request.loss_readings],
                    "shortwave_radiation": [r.ghi_w_m2 for r in request.loss_readings],
                    "temperature_2m": [r.temperature_c for r in request.loss_readings],
                },
                index=timestamps,
            )
            loss_report = attribute_losses(loss_df, plant_config)
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