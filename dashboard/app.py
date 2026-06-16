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
import pickle
import sys
from pathlib import Path
from datetime import datetime

# ── Path setup ────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Zenith — Solar Intelligence Platform",
    page_icon="🌤️",
    layout="wide"
)

# ── Load model ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load XGBoost model — tries multiple paths for local and Docker."""
    possible_paths = [
        Path(__file__).resolve().parent / "src" / "models" / "xgboost_solar_v2.pkl",
        Path("/app/src/models/xgboost_solar_v2.pkl"),
        Path("src/models/xgboost_solar_v2.pkl"),
        Path(__file__).resolve().parent.parent / "src" / "models" / "xgboost_solar_v2.pkl",
    ]

    for path in possible_paths:
        if path.exists():
            try:
                with open(path, "rb") as f:
                    return pickle.load(f), True
            except Exception:
                continue

    return None, False

model, model_loaded = load_model()

# ── Weather fetcher ───────────────────────────────────────────
@st.cache_data(ttl=3600)  # cache for 1 hour
def get_live_weather():
    """
    Fetch real live weather from Open-Meteo for Jaipur.
    Cached for 1 hour — refreshes automatically.
    """
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude"      : 26.9124,
            "longitude"     : 75.7873,
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
def run_forecast(weather_data):
    """Run XGBoost model on weather data."""
    if model is None:
        return None

    records = []
    for h in weather_data:
        hour  = h['hour']
        month = h['month']
        records.append({
            'cloud_cover'          : h['cloud_cover'],
            'shortwave_radiation'  : h['shortwave_radiation'],
            'temperature_2m'       : h['temperature_2m'],
            'relative_humidity_2m' : h['relative_humidity_2m'],
            'wind_speed_10m'       : h['wind_speed_10m'],
            'hour sin'             : np.sin(2 * np.pi * hour / 24),
            'hour cos'             : np.cos(2 * np.pi * hour / 24),
            'month sin'            : np.sin(2 * np.pi * month / 12),
            'month cos'            : np.cos(2 * np.pi * month / 12),
            'solar_lag_1h'         : 0.0,
            'solar_lag_24h'        : 0.0,
            'solar_lag_48h'        : 0.0,
            'solar_lag_168h'       : 0.0,
            'solar_rolling_mean_3h': 0.0,
            'solar_rolling_mean_6h': 0.0,
            'solar_rolling_std_3h' : 0.0,
            'clear_sky_ratio'      : max(0, h['shortwave_radiation'] / 950),
        })

    X = pd.DataFrame(records)
    predictions = model.predict(X)
    return np.clip(predictions, 0, 100).tolist()

# ── Battery optimizer ─────────────────────────────────────────
def run_optimization(solar_forecast, demand_forecast,
                     battery_capacity=50, charge_rate=25,
                     discharge_rate=25, initial_charge=25):
    """
    Simple rule-based battery optimizer.
    No PuLP needed — works on Streamlit Cloud.
    """
    results = []
    battery = initial_charge

    for t in range(len(solar_forecast)):
        solar   = solar_forecast[t]
        demand  = demand_forecast[t]
        surplus = solar - demand

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
            'hour'               : t,
            'solar_mw'           : round(solar, 2),
            'demand_mw'          : round(demand, 2),
            'surplus_mw'         : round(surplus, 2),
            'charge_mw'          : round(charge_amt, 2),
            'discharge_mw'       : round(discharge_amt, 2),
            'battery_level_mwh'  : round(battery, 2),
            'grid_balance_mw'    : round(solar + discharge_amt - charge_amt - demand, 2),
            'action'             : action
        })

    return pd.DataFrame(results)

# ══════════════════════════════════════════════════════════════
# DASHBOARD LAYOUT
# ══════════════════════════════════════════════════════════════

# Header
st.title("⚡ Zenith")
st.caption("Peak solar intelligence for India's grid — Jaipur, Rajasthan")

# Model status
if model_loaded:
    st.success("✅ Model loaded | XGBoost solar forecasting active")
else:
    st.error("❌ Model not found — check src/models/xgboost_solar_v2.pkl")
    st.stop()

# Fetch weather
weather_data, is_live = get_live_weather()

# Live/demo indicator
if is_live:
    st.success(
        f"🌤️ Live weather — Jaipur, Rajasthan | "
        f"Updated: {datetime.now().strftime('%d %b %Y, %I:%M %p IST')}"
    )
else:
    st.info("📊 Live weather unavailable — showing demo forecast")

st.divider()

# ── Run forecast ──────────────────────────────────────────────
predictions = run_forecast(weather_data)

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
col4.metric("Plant Capacity",      "100 MW")

st.divider()

# ── Forecast chart ────────────────────────────────────────────
st.subheader("24-Hour Solar Generation Forecast")

demand = [
    round(40 + 8 * np.sin(np.pi * (h - 6) / 12), 2)
    for h in hours
]

fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=hours, y=predictions,
    name='Solar forecast',
    line=dict(color='orange', width=2),
    fill='tozeroy',
    fillcolor='rgba(255,165,0,0.15)'
))
fig1.add_trace(go.Scatter(
    x=hours, y=demand,
    name='Demand forecast',
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
    predictions, demand,
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