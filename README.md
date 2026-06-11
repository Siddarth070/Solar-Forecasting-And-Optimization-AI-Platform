# Solar-Forecasting-And-Optimization-AI-Platform
End-to-end AI platform for solar energy forecasting and grid optimization  built for Jaipur, Rajasthan. Real weather data + physics-informed ML +  XGBoost/LSTM ensemble → actionable recommendations for grid operators.
# ☀️ AI Solar Energy Forecasting & Grid Optimization Platform

> Helping Indian grid operators predict solar generation and make 
> smarter decisions about storage, backup power, and curtailment.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Status](https://img.shields.io/badge/Status-Active_Development-green)
![Phase](https://img.shields.io/badge/Phase-1_EDA_Complete-orange)

---

## The Problem

India wastes thousands of MWh of clean solar energy every day because 
grid operators cannot reliably predict how much solar power their plants 
will generate. Without accurate forecasts they:

- Keep coal plants running on standby — burning fuel for nothing
- Curtail (switch off) solar plants when the grid can't absorb surplus
- React to supply-demand gaps instead of preparing for them

This costs utilities crores of rupees annually and slows India's 
transition to clean energy.

---

## The Solution

An end-to-end AI platform that forecasts solar generation up to 24 hours 
ahead, detects demand-supply mismatches, and tells grid operators exactly 
what to do — charge batteries, start backup, or absorb surplus — before 
the problem occurs.

---

## Architecture
---

## Who Uses This

| User | Problem solved |
|------|---------------|
| Grid operators (SLDC) | 24-hour solar forecast for grid balancing |
| Utility companies (DISCOM) | Reduce CERC deviation settlement penalties |
| Solar plant owners (IPP) | Accurate scheduling to minimize penalties |
| Battery storage operators | Optimal charge/discharge scheduling |

---

## Project Status

| Phase | What | Status |
|-------|------|--------|
| 0 | Environment · data pipeline · simulator | ✅ Complete |
| 1 | EDA · data quality · pattern analysis | ✅ Complete |
| 2 | Feature engineering | 🔄 In progress |
| 3 | XGBoost + LSTM + Prophet models + MLflow | ⏳ Next |
| 4 | Optimization engine | ⏳ Planned |
| 5 | FastAPI + Streamlit dashboard | ⏳ Planned |
| 6 | Docker + GCP deployment | ⏳ Planned |

---

## Key Findings from EDA (Phase 1)

- **Irradiance is the dominant driver** — correlation of 1.0 with 
  solar output. The single most important feature.
- **Hour of day matters most** — daily bell curve peaks at 11–12 PM 
  solar noon over Jaipur.
- **Seasonal trend confirmed** — output increases ~1.5 MW/month 
  from January toward summer.
- **Cloud cover effect** — real but requires full-year data including 
  monsoon months to measure accurately.
- **Wind speed irrelevant** — 0.05 correlation. Excluded from features.

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Data | Python · Pandas · NumPy · PyArrow |
| Weather API | Open-Meteo · NASA POWER |
| Models | XGBoost · TensorFlow/Keras · Prophet |
| Tracking | MLflow |
| Orchestration | Apache Airflow |
| API | FastAPI |
| Dashboard | Streamlit |
| Infrastructure | Docker · GCP |

---
