# Solar Energy Forecasting & Grid Optimization Platform

An end-to-end AI platform that forecasts solar energy generation and electricity
demand for Jaipur, Rajasthan — enabling grid operators to make proactive decisions
about battery storage, backup activation, and load management.

---

## The Problem

India wastes thousands of MWh of clean solar energy daily because grid operators
cannot predict generation accurately enough to make smart decisions. Poor forecasting
leads to unnecessary curtailment, idle backup fossil fuel burning, and grid instability.

## The Solution

This platform uses a three-model ensemble (XGBoost + LSTM + Prophet) with quantile
regression to deliver 15-minute, 1-hour, and 24-hour solar forecasts with uncertainty
bounds. The optimization engine converts forecasts into actionable recommendations:
when to charge batteries, when to activate backup, when surplus can be absorbed.

---

## Architecture

```
Weather APIs (Open-Meteo, NASA POWER)
         ↓
  Data Ingestion Pipeline (Airflow DAGs)
         ↓
  Feature Engineering (lag features, clear-sky ratio, cyclical encoding)
         ↓
  Forecasting Models (XGBoost + LSTM + Prophet ensemble)
         ↓
  Optimization Engine (storage dispatch, curtailment reduction)
         ↓
  FastAPI serving layer
         ↓
  Streamlit dashboard → grid operators
```

---

## Project Structure

```
solar_forecast_platform/
├── configs/            # All configuration — nothing hardcoded in code
├── data/
│   ├── raw/            # Original API responses — never modified
│   ├── processed/      # Cleaned, validated data
│   └── features/       # Feature-engineered tables ready for modelling
├── src/
│   ├── ingestion/      # API fetchers + data validators
│   ├── features/       # Feature engineering pipeline
│   ├── models/         # XGBoost, LSTM, Prophet, ensemble
│   ├── optimization/   # Grid dispatch logic
│   └── api/            # FastAPI endpoints
├── airflow/dags/       # Orchestration — nightly retraining, daily ingestion
├── dashboard/          # Streamlit operator dashboard
├── tests/              # Unit + integration tests
└── docs/               # Architecture decisions, API docs
```

---

## Quickstart

```bash
# Clone and set up
git clone https://github.com/YOUR_USERNAME/solar-forecast-platform.git
cd solar-forecast-platform

# Install dependencies
pip install -r requirements.txt

# Generate simulated Jaipur data
python src/ingestion/jaipur_simulator.py

# Run tests
pytest tests/ -v
```

---

## Development Roadmap

| Phase | Focus | Status |
|-------|-------|--------|
| 0 | Environment setup + data ingestion | ✅ Complete |
| 1 | EDA + data pipeline | 🔄 Next |
| 2 | Feature engineering | ⏳ Planned |
| 3 | Forecasting models + MLflow | ⏳ Planned |
| 4 | Optimization engine | ⏳ Planned |
| 5 | FastAPI + Streamlit dashboard | ⏳ Planned |
| 6 | Docker + GCP deployment | ⏳ Planned |

---

## Tech Stack

**Data:** Python, Pandas, NumPy, PyArrow
**Models:** XGBoost, TensorFlow/Keras (LSTM), Prophet
**Tracking:** MLflow
**Orchestration:** Apache Airflow
**API:** FastAPI
**Dashboard:** Streamlit
**Infrastructure:** Docker, GCP

---

## Target Users

- **State Load Dispatch Centres (SLDCs)** — 24-hour generation forecasts for grid balancing
- **DISCOMs** — Reduce deviation settlement penalties
- **Solar IPPs** — Accurate scheduling to minimise CERC penalties
- **Battery storage operators** — Optimal charge/discharge scheduling

---

## Data Sources

- [Open-Meteo](https://open-meteo.com) — Free hourly weather forecasts + historical archive
- [NASA POWER](https://power.larc.nasa.gov) — Satellite-based solar irradiance data
- [POSOCO](https://posoco.in) — Indian grid load data

---

*Built as a portfolio project demonstrating end-to-end ML engineering for the Indian
energy sector. Target location: Jaipur, Rajasthan (one of India's highest solar
irradiance regions, ~5.5–6.0 kWh/m²/day annual GHI).*


## Live Demo
👉 [Open Dashboard](https://zenith-to.streamlit.app/)
