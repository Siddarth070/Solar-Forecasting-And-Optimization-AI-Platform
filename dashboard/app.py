
import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Solar Forecast Platform — Jaipur",
    page_icon="☀️",
    layout="wide"
)

# ── API connection ─────────────────────────────────────────────
API_URL = "http://127.0.0.1:8000"

# ── Helper functions ──────────────────────────────────────────

def check_api():
    """Check if API is running."""
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.json()
    except:
        return None

def generate_demo_weather(hours=24):
    """
    Generate realistic demo weather for Jaipur.
    Used when no live API data is available.
    """
    hour_range = np.arange(hours)
    solar_noon = 12.5

    hour_angle   = (hour_range % 24 - solar_noon) * (np.pi / 12)
    solar_factor = np.maximum(0, np.cos(hour_angle))
    ghi          = 750 * solar_factor + np.random.normal(0, 20, hours)
    ghi          = np.clip(ghi, 0, 900)

    weather = []
    for i, h in enumerate(hour_range):
        weather.append({
            "shortwave_radiation"  : round(float(ghi[i]), 1),
            "cloud_cover"          : round(float(np.random.uniform(5, 25)), 1),
            "temperature_2m"       : round(float(25 + 10 * solar_factor[i]), 1),
            "relative_humidity_2m" : round(float(40 - 10 * solar_factor[i]), 1),
            "wind_speed_10m"       : round(float(np.random.uniform(2, 6)), 1),
            "hour"                 : int(h % 24),
            "month"                : 3
        })
    return weather

def get_forecast(weather_data):
    """Call forecast API endpoint."""
    try:
        r = requests.post(
            f"{API_URL}/forecast",
            json={"hours": weather_data, "plant_capacity_mw": 100},
            timeout=10
        )
        return r.json()
    except Exception as e:
        st.error(f"Forecast API error: {e}")
        return None

def get_optimization(solar_forecast, demand_forecast):
    """Call optimization API endpoint."""
    try:
        r = requests.post(
            f"{API_URL}/optimize",
            json={
                "solar_forecast_mw" : solar_forecast,
                "demand_forecast_mw": demand_forecast,
            },
            timeout=10
        )
        return r.json()
    except Exception as e:
        st.error(f"Optimization API error: {e}")
        return None


# ── Dashboard layout ──────────────────────────────────────────

# Header
st.title("☀️ Solar Forecast Platform")
st.caption("Jaipur, Rajasthan — AI-based generation forecasting and grid optimization")

# API status
api_status = check_api()
if api_status:
    st.success(f"✅ API connected | Model loaded: {api_status['model_loaded']}")
else:
    st.error("❌ API not connected — start FastAPI server first")
    st.code("uvicorn src.api.main:app --reload --port 8000")
    st.stop()

st.divider()

# ── Section 1 — KPI cards ──────────────────────────────────────
st.subheader("Today's Overview")

weather  = generate_demo_weather(24)
forecast = get_forecast(weather)

if forecast and 'predictions_mw' in forecast:
    predictions = forecast['predictions_mw']
    total_gen   = forecast['total_generation_mwh']
    peak_mw     = forecast['peak_output_mw']
    peak_hour   = forecast['peak_hour']

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Expected Generation",
        f"{total_gen:.0f} MWh",
        help="Total solar energy expected today"
    )
    col2.metric(
        "Peak Output",
        f"{peak_mw:.1f} MW",
        help="Maximum output expected today"
    )
    col3.metric(
        "Peak Hour",
        f"{peak_hour:02d}:00",
        help="Hour of maximum generation"
    )
    col4.metric(
        "Plant Capacity",
        "100 MW",
        help="Installed capacity of Jaipur solar plant"
    )

st.divider()

# ── Section 2 — Forecast chart ─────────────────────────────────
st.subheader("24-Hour Solar Generation Forecast")

if forecast:
    hours  = list(range(24))
    demand = [
        40 + 8 * np.sin(np.pi * (h - 6) / 12)
        for h in hours
    ]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=hours, y=predictions,
        name='Solar forecast',
        line=dict(color='orange', width=2),
        fill='tozeroy',
        fillcolor='rgba(255,165,0,0.1)'
    ))

    fig.add_trace(go.Scatter(
        x=hours, y=demand,
        name='Demand forecast',
        line=dict(color='royalblue', width=2, dash='dash')
    ))

    fig.update_layout(
        xaxis_title='Hour of day',
        yaxis_title='MW',
        hovermode='x unified',
        legend=dict(orientation='h', y=1.1),
        height=350
    )

    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Section 3 — Optimization ───────────────────────────────────
st.subheader("Battery Dispatch Recommendations")

if forecast:
    demand_forecast = [
        round(40 + 8 * np.sin(np.pi * (h - 6) / 12), 2)
        for h in range(24)
    ]

    opt = get_optimization(predictions, demand_forecast)

    if opt:
        schedule = pd.DataFrame(opt['schedule'])
        summary  = opt['summary']

        # Summary metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Charged",    f"{summary['total_charged_mwh']} MWh")
        c2.metric("Total Discharged", f"{summary['total_discharged_mwh']} MWh")
        c3.metric("Hours Active",
                  f"{summary['hours_charging'] + summary['hours_discharging']}h")

        # Battery level chart
        fig2 = go.Figure()

        fig2.add_trace(go.Scatter(
            x=list(range(len(schedule))),
            y=schedule['battery_level_mwh'],
            name='Battery level',
            line=dict(color='purple', width=2),
            fill='tozeroy',
            fillcolor='rgba(128,0,128,0.1)'
        ))

        fig2.add_hline(
            y=50, line_dash='dash',
            line_color='red',
            annotation_text='Max capacity (50 MWh)'
        )

        fig2.update_layout(
            xaxis_title='Hour',
            yaxis_title='MWh',
            height=300
        )

        st.plotly_chart(fig2, use_container_width=True)

        # Action recommendations table
        st.subheader("Hour-by-Hour Recommendations")

        def color_action(val):
            if val == 'CHARGE':
                return 'background-color: #d4edda'
            elif val == 'DISCHARGE':
                return 'background-color: #f8d7da'
            return ''

        display_df = schedule[[
            'solar_mw', 'demand_mw', 'surplus_mw',
            'charge_mw', 'discharge_mw',
            'battery_level_mwh', 'action'
        ]].copy()

        styled = display_df.style.map(color_action, subset=['action'])
        st.dataframe(
            styled,
            use_container_width=True,
            height=400
        )

st.divider()

# ── Section 4 — Alerts ─────────────────────────────────────────
st.subheader("Smart Alerts")

if forecast and opt:
    schedule_df = pd.DataFrame(opt['schedule'])

    # Find shortage hours
    shortage_hours = schedule_df[
        schedule_df['grid_balance_mw'] < -5
    ]['grid_balance_mw']

    if len(shortage_hours) > 0:
        st.warning(
            f"⚠️ {len(shortage_hours)} hours with grid shortage detected. "
            f"Maximum shortage: {shortage_hours.min():.1f} MW. "
            f"Consider activating backup power."
        )

    # Find peak surplus
    max_surplus = schedule_df['surplus_mw'].max()
    if max_surplus > 10:
        peak_surplus_hour = schedule_df['surplus_mw'].idxmax()
        st.info(
            f"💡 Peak surplus of {max_surplus:.1f} MW expected at "
            f"hour {peak_surplus_hour:02d}:00. "
            f"Battery charging recommended."
        )

    # Battery low warning
    min_battery = schedule_df['battery_level_mwh'].min()
    if min_battery < 5:
        st.warning(
            f"🔋 Battery level drops to {min_battery:.1f} MWh. "
            f"Consider reducing discharge rate or increasing capacity."
        )

    st.success("✅ Optimization complete — dispatch schedule ready")

# Footer
st.divider()
st.caption(
    "Solar Forecast Platform v1.0 | "
    "Built by Siddharth Agrawal | "
    "Jaipur, Rajasthan, India"
)