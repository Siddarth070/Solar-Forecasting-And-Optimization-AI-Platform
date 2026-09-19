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

The deployed dashboard runs as a **standalone Streamlit app** — it loads the
XGBoost model and calls Open-Meteo directly, with no FastAPI hop in between.
A separate FastAPI service also exists (`src/api/main.py`) for programmatic
access, but it is not part of the live deployment path.

```
Open-Meteo (live weather)
         ↓
  Standalone Streamlit dashboard
    ├── loads XGBoost model directly (src/models/xgboost_solar_v2.pkl)
    └── runs a rule-based battery dispatch optimizer inline
         ↓
  Grid operator view (KPIs, forecast chart, dispatch schedule, alerts)
```

Model training (XGBoost, LSTM, Prophet, ensemble) happens in the notebooks
under `notebooks/` — there is no `src/` pipeline yet to retrain models from
code (see Known Gaps below).

---

## Project Structure

```
.
├── configs/            # All configuration — nothing hardcoded in code
├── src/
│   ├── ingestion/      # Open-Meteo fetcher + Jaipur data simulator
│   ├── features/       # Placeholder — feature engineering currently lives in notebooks/
│   ├── models/         # Serialized XGBoost model used by the dashboard/API
│   ├── optimization/   # PuLP-based battery dispatch optimizer (not yet wired into the live dashboard)
│   └── api/            # FastAPI service (standalone — not used by the deployed dashboard)
├── dashboard/           # Standalone Streamlit app — this is what's deployed
├── notebooks/          # EDA, feature engineering, model training (XGBoost/LSTM/Prophet/ensemble/optimization)
└── tests/              # Unit tests
```

---

## Quickstart

```bash
# Clone and set up
git clone https://github.com/siddarth070/solar-forecasting-and-optimization-ai-platform.git
cd solar-forecasting-and-optimization-ai-platform

# Install dependencies
pip install -r requirements.txt

# Run the dashboard locally
streamlit run dashboard/app.py

# Run tests
pytest tests/ -v
```

---

## Development Status

| Phase | Focus | Status |
|-------|-------|--------|
| 0 | Environment setup + data ingestion | ✅ Complete |
| 1 | EDA + data pipeline | ✅ Complete |
| 2 | Feature engineering | ✅ Complete (in notebooks) |
| 3 | Forecasting models (XGBoost, LSTM, Prophet) | ✅ Complete (in notebooks) |
| 4 | Optimization engine | ✅ Complete (`src/optimization/`, not yet wired into the dashboard) |
| 5 | FastAPI + Streamlit dashboard | ✅ Complete |
| 6 | Docker + live deployment | ✅ Complete — [live demo](https://zenith-to.streamlit.app/) |

### Known Gaps

- `src/features/` has no code yet — feature engineering only exists in
  `notebooks/feature_engineering.ipynb`.
- The deployed dashboard uses a simple rule-based optimizer inline, not the
  PuLP-based `src/optimization/battery_optimizer.py`.
- `src/api/main.py` (FastAPI) isn't used by the live deployment; it's kept
  for programmatic/API access.
- Only the XGBoost model is wired into production; LSTM and Prophet exist as
  trained artifacts under `notebooks/src/models/` but aren't served.

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
