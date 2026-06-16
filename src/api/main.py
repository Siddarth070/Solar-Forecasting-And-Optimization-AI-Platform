import numpy as np
import pandas as pd
import pickle
import sys
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from loguru import logger

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.optimization.battery_optimizer import BatteryOptimizer

# ── App setup ─────────────────────────────────────────────────
app = FastAPI(
    title="Solar Forecast Platform",
    description="AI-based solar energy forecasting and grid optimization for Jaipur, Rajasthan",
    version="1.0.0"
)

# ── Load model on startup ─────────────────────────────────────
MODEL_PATH = PROJECT_ROOT / "src" / "models" / "xgboost_solar_v2.pkl"

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    logger.success(f"Model loaded from {MODEL_PATH}")
except FileNotFoundError:
    logger.error(f"Model not found at {MODEL_PATH}")
    model = None


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
    plant_capacity_mw: float = Field(
        default=100.0,
        description="Solar plant capacity in MW"
    )


class ForecastResponse(BaseModel):
    """Response from forecast endpoint."""
    forecast_hours     : int
    predictions_mw     : list[float]
    peak_output_mw     : float
    peak_hour          : int
    total_generation_mwh: float
    generated_at       : str


class OptimizeRequest(BaseModel):
    """Request body for optimization endpoint."""
    solar_forecast_mw : list[float] = Field(..., description="Hourly solar forecast")
    demand_forecast_mw: list[float] = Field(..., description="Hourly demand forecast")
    battery_capacity_mwh : float = Field(default=50.0)
    charge_rate_mw       : float = Field(default=25.0)
    discharge_rate_mw    : float = Field(default=25.0)
    initial_charge_mwh   : float = Field(default=25.0)


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


@app.post("/forecast", response_model=ForecastResponse)
def forecast(request: ForecastRequest):
    """
    Generate solar output forecast from weather inputs.

    Takes hourly weather data and returns predicted
    solar generation for each hour.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check server logs."
        )

    try:
        # Build feature DataFrame from request
        records = []
        for h in request.hours:
            hour_sin  = np.sin(2 * np.pi * h.hour / 24)
            hour_cos  = np.cos(2 * np.pi * h.hour / 24)
            month_sin = np.sin(2 * np.pi * h.month / 12)
            month_cos = np.cos(2 * np.pi * h.month / 12)

            records.append({
                'cloud_cover': h.cloud_cover,
                'shortwave_radiation': h.shortwave_radiation,
                'temperature_2m': h.temperature_2m,
                'relative_humidity_2m': h.relative_humidity_2m,
                'wind_speed_10m': h.wind_speed_10m,
                'hour sin': np.sin(2 * np.pi * h.hour / 24),
                'hour cos': np.cos(2 * np.pi * h.hour / 24),
                'month sin': np.sin(2 * np.pi * h.month / 12),
                'month cos': np.cos(2 * np.pi * h.month / 12),
                'solar_lag_1h': 0.0,
                'solar_lag_24h': 0.0,
                'solar_lag_48h': 0.0,
                'solar_lag_168h': 0.0,
                'solar_rolling_mean_3h': 0.0,
                'solar_rolling_mean_6h': 0.0,
                'solar_rolling_std_3h': 0.0,
                'clear_sky_ratio': max(0, h.shortwave_radiation / 950),
            }),


        X = pd.DataFrame(records)
        print("Columns being sent to model:", list(X.columns))
        print("Model expects:", model.get_booster().feature_names)
        predictions = model.predict(X)
        predictions = np.clip(
            predictions, 0, request.plant_capacity_mw
        ).tolist()

        peak_idx = int(np.argmax(predictions))

        logger.info(
            f"Forecast generated: {len(predictions)} hours, "
            f"peak {max(predictions):.1f} MW at hour {peak_idx}"
        )

        return ForecastResponse(
            forecast_hours      = len(predictions),
            predictions_mw      = [round(p, 2) for p in predictions],
            peak_output_mw      = round(max(predictions), 2),
            peak_hour           = peak_idx,
            total_generation_mwh= round(sum(predictions), 2),
            generated_at        = datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Forecast error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/optimize")
def optimize(request: OptimizeRequest):
    """
    Run battery dispatch optimization.

    Takes solar and demand forecasts, returns
    optimal charge/discharge schedule.
    """
    try:
        optimizer = BatteryOptimizer(
            battery_capacity_mwh = request.battery_capacity_mwh,
            charge_rate_mw       = request.charge_rate_mw,
            discharge_rate_mw    = request.discharge_rate_mw,
            initial_charge_mwh   = request.initial_charge_mwh,
        )

        results = optimizer.optimize(
            solar_forecast  = np.array(request.solar_forecast_mw),
            demand_forecast = np.array(request.demand_forecast_mw),
        )

        schedule_records = []
        for record in results.to_dict(orient="records"):
            clean = {k: float(v) if hasattr(v, 'item') else v
                     for k, v in record.items()}
            schedule_records.append(clean)

        return {
            "status": "optimal",
            "hours": int(len(results)),
            "schedule": schedule_records,
            "summary": {
                "total_charged_mwh": float(round(results['charge_mw'].sum(), 2)),
                "total_discharged_mwh": float(round(results['discharge_mw'].sum(), 2)),
                "hours_charging": int((results['action'] == 'CHARGE').sum()),
                "hours_discharging": int((results['action'] == 'DISCHARGE').sum()),
                "hours_hold": int((results['action'] == 'HOLD').sum()),
            }
        }
    except Exception as e:
        logger.error(f"Optimization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))