"""
app.py — Standalone Solar Forecast Dashboard
---------------------------------------------
Runs completely independently — no FastAPI needed.
Loads XGBoost model and optimizer directly.
Pulls live weather from Open-Meteo API.
Deployable to Streamlit Cloud with zero configuration.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import sys
from pathlib import Path
from datetime import datetime
from xgboost import XGBRegressor

# ── Path setup ────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT.parent))

from src.features.pipeline import build_features, SERVING_FEATURE_COLUMNS
from src.utils.config_loader import get_plant_config, list_plant_ids

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Zenith — Solar Intelligence Platform",
    page_icon="🌤️",
    layout="wide"
)

# ── Plant selection (roadmap P1.1 — no hardcoded single location) ──
plant_ids = list_plant_ids()
with st.sidebar:
    st.header("🏭 Plant")
    selected_plant_id = st.selectbox(
        "Select plant", plant_ids,
        format_func=lambda pid: get_plant_config(pid)["name"],
    )
PLANT = get_plant_config(selected_plant_id)
PLANT_CAPACITY_MW = PLANT["capacity"]["ac_capacity_mw"]

# ── Load model ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load XGBoost model — tries multiple paths for local and Docker.
    JSON, not pickle: a pickle can silently break across library versions
    and is opaque to review (roadmap P0.6)."""
    possible_paths = [
        Path(__file__).resolve().parent / "src" / "models" / "xgboost_solar_v2.json",
        Path("/app/src/models/xgboost_solar_v2.json"),
        Path("src/models/xgboost_solar_v2.json"),
        Path(__file__).resolve().parent.parent / "src" / "models" / "xgboost_solar_v2.json",
    ]

    for path in possible_paths:
        if path.exists():
            try:
                model = XGBRegressor()
                model.load_model(str(path))
                return model, True
            except Exception:
                continue

    return None, False

model, model_loaded = load_model()

@st.cache_resource
def load_quantile_model():
    """Load the P10/P50/P90 quantile model (roadmap P1.3). Optional -- the
    point forecast above works without it; its absence just means no
    uncertainty band is shown."""
    possible_paths = [
        Path(__file__).resolve().parent / "src" / "models" / "xgboost_solar_quantile.json",
        Path("/app/src/models/xgboost_solar_quantile.json"),
        Path("src/models/xgboost_solar_quantile.json"),
        Path(__file__).resolve().parent.parent / "src" / "models" / "xgboost_solar_quantile.json",
    ]
    for path in possible_paths:
        if path.exists():
            try:
                qmodel = XGBRegressor()
                qmodel.load_model(str(path))
                return qmodel, True
            except Exception:
                continue
    return None, False

quantile_model, quantile_model_loaded = load_quantile_model()

# ── Weather fetcher ───────────────────────────────────────────
@st.cache_data(ttl=3600)  # cache for 1 hour, per (latitude, longitude)
def get_live_weather(latitude: float, longitude: float):
    """
    Fetch real live weather from Open-Meteo for the given coordinates.
    Cached for 1 hour — refreshes automatically.
    """
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude"      : latitude,
            "longitude"     : longitude,
            "hourly"        : [
                "temperature_2m",
                "relative_humidity_2m",
                "cloud_cover",
                "wind_speed_10m",
                "shortwave_radiation"
            ],
            "timezone"      : "Asia/Kolkata",
            "forecast_days" : 1
        }
        r = requests.get(url, params=params, timeout=10)
        data = r.json()['hourly']

        records = []
        for i in range(24):
            records.append({
                "shortwave_radiation"  : float(data['shortwave_radiation'][i] or 0),
                "cloud_cover"          : float(data['cloud_cover'][i] or 0),
                "temperature_2m"       : float(data['temperature_2m'][i] or 25),
                "relative_humidity_2m" : float(data['relative_humidity_2m'][i] or 40),
                "wind_speed_10m"       : float(data['wind_speed_10m'][i] or 3),
                "hour"                 : i,
                "month"                : datetime.now().month
            })
        return records, True

    except Exception as e:
        # Fallback to demo weather
        hour_range = np.arange(24)
        solar_noon = 12.5
        hour_angle   = (hour_range - solar_noon) * (np.pi / 12)
        solar_factor = np.maximum(0, np.cos(hour_angle))
        ghi          = 750 * solar_factor

        records = []
        for i in range(24):
            records.append({
                "shortwave_radiation"  : round(float(ghi[i]), 1),
                "cloud_cover"          : 15.0,
                "temperature_2m"       : round(float(25 + 10 * solar_factor[i]), 1),
                "relative_humidity_2m" : 40.0,
                "wind_speed_10m"       : 3.5,
                "hour"                 : i,
                "month"                : datetime.now().month
            })
        return records, False

# ── Forecast function ─────────────────────────────────────────
def run_forecast(weather_data, plant_config, capacity_mw):
    """Run XGBoost model on weather data via the canonical feature pipeline
    (src/features/pipeline.py) — no more hand-built, drifting feature dict."""
    if model is None:
        return None

    # build_features needs a real timestamp (for the solar-position
    # clear-sky estimate, using THIS plant's lat/lon); weather_data only
    # carries hour + month, so we anchor to a fixed reference year/day,
    # same approach as the API (src/api/main.py) — an approximation
    # pending a proper per-timestamp request shape (P1.2).
    timestamps = pd.DatetimeIndex([
        pd.Timestamp(year=2024, month=h["month"], day=15, hour=h["hour"],
                     tz="Asia/Kolkata")
        for h in weather_data
    ])
    raw = pd.DataFrame(weather_data, index=timestamps)
    features = build_features(raw, plant_config)
    X = features[SERVING_FEATURE_COLUMNS]

    # Model outputs capacity FRACTION (0-1), not MW (see
    # src/models/train.py) -- rescale by THIS plant's own capacity so one
    # model correctly serves differently-sized plants.
    predictions = model.predict(X) * capacity_mw
    return np.clip(predictions, 0, capacity_mw).tolist()

# ── Probabilistic forecast (P10/P50/P90) ─────────────────────────
def run_quantile_forecast(weather_data, plant_config, capacity_mw):
    """Same feature pipeline as run_forecast, but through the quantile
    model (roadmap P1.3). Returns None if the quantile model isn't
    available -- callers must handle that gracefully, not crash."""
    if quantile_model is None:
        return None

    timestamps = pd.DatetimeIndex([
        pd.Timestamp(year=2024, month=h["month"], day=15, hour=h["hour"],
                     tz="Asia/Kolkata")
        for h in weather_data
    ])
    raw = pd.DataFrame(weather_data, index=timestamps)
    features = build_features(raw, plant_config)
    X = features[SERVING_FEATURE_COLUMNS]

    # Three columns [P10, P50, P90], capacity fraction -- rescale, then
    # sort defensively so a rare crossing never yields P10 > P90.
    q_preds = np.clip(quantile_model.predict(X) * capacity_mw, 0, capacity_mw)
    q_preds = np.sort(q_preds, axis=1)
    return q_preds[:, 0].tolist(), q_preds[:, 1].tolist(), q_preds[:, 2].tolist()

# ── Battery optimizer ─────────────────────────────────────────
def run_optimization(solar_forecast, declared_schedule_mw,
                     battery_capacity=50, charge_rate=25,
                     discharge_rate=25, initial_charge=25):
    """
    Simple rule-based battery optimizer.
    No PuLP needed — works on Streamlit Cloud.

    declared_schedule_mw: what the plant committed to deliver to the grid
    each hour — a real IPP has a schedule, not a "demand" curve to guess
    at (roadmap P1.6). Set by the operator (see the sidebar), never
    fabricated by this function.
    """
    results = []
    battery = initial_charge

    for t in range(len(solar_forecast)):
        solar   = solar_forecast[t]
        schedule = declared_schedule_mw[t]
        surplus = solar - schedule

        charge_amt    = 0.0
        discharge_amt = 0.0

        if surplus > 0:
            charge_amt = min(surplus, charge_rate,
                           battery_capacity - battery)
            battery   += charge_amt
            action     = "CHARGE" if charge_amt > 0.5 else "HOLD"
        elif surplus < 0:
            discharge_amt = min(abs(surplus), discharge_rate, battery)
            battery      -= discharge_amt
            action        = "DISCHARGE" if discharge_amt > 0.5 else "HOLD"
        else:
            action = "HOLD"

        results.append({
            'hour'                    : t,
            'solar_mw'                : round(solar, 2),
            'declared_schedule_mw'    : round(schedule, 2),
            'surplus_mw'              : round(surplus, 2),
            'charge_mw'               : round(charge_amt, 2),
            'discharge_mw'            : round(discharge_amt, 2),
            'battery_level_mwh'       : round(battery, 2),
            'grid_balance_mw'         : round(solar + discharge_amt - charge_amt - schedule, 2),
            'action'                  : action
        })

    return pd.DataFrame(results)

# ══════════════════════════════════════════════════════════════
# DASHBOARD LAYOUT
# ══════════════════════════════════════════════════════════════

# Header
st.title("⚡ Zenith")
st.caption(
    f"Peak solar intelligence for India's grid — "
    f"{PLANT['location']['name']}, {PLANT['location']['state']}"
)

# Model status
if model_loaded:
    st.success("✅ Model loaded | XGBoost solar forecasting active")
else:
    st.error("❌ Model not found — check src/models/xgboost_solar_v2.json")
    st.stop()

# Fetch weather
weather_data, is_live = get_live_weather(
    PLANT["location"]["latitude"], PLANT["location"]["longitude"]
)

# Live/demo indicator
if is_live:
    st.success(
        f"🌤️ Live weather — {PLANT['location']['name']}, {PLANT['location']['state']} | "
        f"Updated: {datetime.now().strftime('%d %b %Y, %I:%M %p IST')}"
    )
else:
    st.info("📊 Live weather unavailable — showing demo forecast")

st.divider()

# ── Run forecast ──────────────────────────────────────────────
predictions = run_forecast(weather_data, PLANT, PLANT_CAPACITY_MW)

if predictions is None:
    st.error("Forecast failed — model not loaded")
    st.stop()

hours       = list(range(24))
total_gen   = round(sum(predictions), 1)
peak_mw     = round(max(predictions), 1)
peak_hour   = int(np.argmax(predictions))

# ── KPI cards ─────────────────────────────────────────────────
st.subheader("Today's Overview")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Expected Generation", f"{total_gen} MWh")
col2.metric("Peak Output",         f"{peak_mw} MW")
col3.metric("Peak Hour",           f"{peak_hour:02d}:00")
col4.metric("Plant Capacity",      f"{PLANT_CAPACITY_MW:.0f} MW")

st.divider()

# ── Forecast chart ────────────────────────────────────────────
st.subheader("24-Hour Solar Generation Forecast")

# Declared schedule: what THIS plant committed to deliver to the grid --
# not a "demand" curve. A solar IPP doesn't have demand, it has a schedule
# (roadmap P1.6). Previously this was a hardcoded 40 + 8*sin(...) formula
# fabricated with no relationship to any real plant; there is no synthetic
# demand anywhere in this dashboard now -- the operator sets their own
# number below.
with st.sidebar:
    st.header("📋 Declared Schedule")
    declared_schedule_mw = st.slider(
        "Flat declared schedule (MW)", 0, int(PLANT_CAPACITY_MW),
        int(PLANT_CAPACITY_MW * 0.4),
        help="What this plant committed to deliver to the grid for every "
             "hour today. A single flat value for now — per-hour schedules "
             "are roadmap P1.5/P2.7."
    )
declared_schedule = [declared_schedule_mw] * len(hours)

quantiles = run_quantile_forecast(weather_data, PLANT, PLANT_CAPACITY_MW)

fig1 = go.Figure()
if quantiles is not None:
    p10, p50, p90 = quantiles
    # P10-P90 band drawn first (roadmap P1.3): P90 as the visible boundary,
    # then P10 filled back down to it -- the standard two-trace band trick,
    # since Plotly only fills between consecutive traces.
    fig1.add_trace(go.Scatter(
        x=hours, y=p90, name='P90', mode='lines',
        line=dict(width=0), showlegend=False, hoverinfo='skip',
    ))
    fig1.add_trace(go.Scatter(
        x=hours, y=p10, name='P10-P90 range', mode='lines',
        line=dict(width=0), fill='tonexty', fillcolor='rgba(255,165,0,0.15)',
    ))
fig1.add_trace(go.Scatter(
    x=hours, y=predictions,
    name='Solar forecast',
    line=dict(color='orange', width=2),
    fill='tozeroy' if quantiles is None else None,
    fillcolor='rgba(255,165,0,0.15)' if quantiles is None else None,
))
fig1.add_trace(go.Scatter(
    x=hours, y=declared_schedule,
    name='Declared schedule',
    line=dict(color='royalblue', width=2, dash='dash')
))
fig1.update_layout(
    xaxis_title='Hour of day',
    yaxis_title='MW',
    hovermode='x unified',
    legend=dict(orientation='h', y=1.1),
    height=350
)
st.plotly_chart(fig1, use_container_width=True)

st.divider()

# ── Optimization ──────────────────────────────────────────────
st.subheader("Battery Dispatch Recommendations")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Battery Settings")
    battery_capacity = st.slider("Battery capacity (MWh)", 10, 100, 50)
    charge_rate      = st.slider("Max charge rate (MW)",   5,  50,  25)
    discharge_rate   = st.slider("Max discharge rate (MW)", 5, 50,  25)
    initial_charge   = st.slider("Initial charge (MWh)",   0,  50,  25)

schedule = run_optimization(
    predictions, declared_schedule,
    battery_capacity, charge_rate,
    discharge_rate, initial_charge
)

# Summary metrics
c1, c2, c3 = st.columns(3)
c1.metric("Total Charged",
          f"{schedule['charge_mw'].sum():.1f} MWh")
c2.metric("Total Discharged",
          f"{schedule['discharge_mw'].sum():.1f} MWh")
c3.metric("Hours Active",
          f"{(schedule['action'] != 'HOLD').sum()}h")

# Battery level chart
fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=schedule['hour'],
    y=schedule['battery_level_mwh'],
    name='Battery level',
    line=dict(color='purple', width=2),
    fill='tozeroy',
    fillcolor='rgba(128,0,128,0.15)'
))
fig2.add_hline(
    y=battery_capacity,
    line_dash='dash', line_color='red',
    annotation_text=f'Max capacity ({battery_capacity} MWh)'
)
fig2.update_layout(
    xaxis_title='Hour',
    yaxis_title='MWh',
    height=280
)
st.plotly_chart(fig2, use_container_width=True)

# Action table
st.subheader("Hour-by-Hour Recommendations")

def color_action(val):
    if val == 'CHARGE':
        return 'background-color: #d4edda; color: black'
    elif val == 'DISCHARGE':
        return 'background-color: #f8d7da; color: black'
    return ''

styled = schedule.style.map(color_action, subset=['action'])
st.dataframe(styled, use_container_width=True, height=400)

st.divider()

# ── Smart alerts ──────────────────────────────────────────────
st.subheader("Smart Alerts")

shortage_hours = schedule[schedule['grid_balance_mw'] < -5]
if len(shortage_hours) > 0:
    st.warning(
        f"⚠️ {len(shortage_hours)} hours with grid shortage. "
        f"Max shortage: {shortage_hours['grid_balance_mw'].min():.1f} MW. "
        f"Consider activating backup power."
    )

max_surplus = schedule['surplus_mw'].max()
if max_surplus > 5:
    peak_surplus_hour = schedule['surplus_mw'].idxmax()
    st.info(
        f"💡 Peak surplus of {max_surplus:.1f} MW at "
        f"hour {peak_surplus_hour:02d}:00. "
        f"Battery charging recommended."
    )

min_battery = schedule['battery_level_mwh'].min()
if min_battery < 5:
    st.warning(
        f"🔋 Battery drops to {min_battery:.1f} MWh. "
        f"Consider increasing capacity."
    )

st.success("✅ Optimization complete — dispatch schedule ready")

# Footer
st.caption(
    f"Zenith v1.0 | "
    f"Built by Siddharth Agrawal | "
)