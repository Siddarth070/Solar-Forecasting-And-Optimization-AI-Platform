# Solar Energy Forecasting & Grid Optimization Platform ("Zenith")

An AI platform that forecasts solar generation for a plant and turns that
forecast into a battery dispatch schedule. Currently trained and validated
on synthetic data for a simulated 100 MW plant in Jaipur, Rajasthan — real
data has not been onboarded yet (see Status below).

This is not the first solar forecasting product in the Indian market —
Renewalytics, ReConnect Energy, EnerMAN, Manikaran Analytics, utility
platforms like Tata Power's CREAMS, and others already operate here. The
intended differentiation is forecast-to-action for 5–100 MW operators and
loss attribution, not primacy.

---

## The Problem

India curtails clean solar generation because grid operators and plant
owners cannot predict generation accurately enough, or act on that
prediction fast enough, to avoid it. Poor forecasting drives unnecessary
curtailment, idle backup capacity, and DSM/deviation penalties.

## What Actually Exists Today

- A single XGBoost model producing an **hourly point forecast** of solar
  output (no ensemble, no quantile/uncertainty bands, no 15-minute
  granularity yet — those are aspirational, not built).
- A feature pipeline (`src/features/pipeline.py`) shared by training, the
  API, and the dashboard, so all three compute features identically.
- A battery dispatch step: the deployed dashboard uses a simple rule-based
  charge/discharge heuristic; a PuLP-based linear-program optimizer exists
  in `src/optimization/` but is not wired into the live dashboard and has
  known modeling gaps (no round-trip efficiency, hourly-only blocks — see
  Known Gaps).
- A reproducible training + evaluation pipeline (`make train`,
  `benchmark.py`) — see Honest Results below for what it actually measures.

---

## Honest Results (synthetic data only)

Scored with `benchmark.py`: nMAE/nRMSE as a percentage of the plant's AC
capacity (never MAPE, which is undefined at dawn/dusk), against three
zero-learned-parameter baselines, on a holdout period the model never
trained on.

| | nMAE |
|---|---|
| **Model** (fed realistic D-1 forecast-quality weather) | **0.64%** |
| Smart persistence (yesterday's clear-sky ratio) | 0.81% |
| Persistence (repeat yesterday) | 0.82% |
| Pure physics (GHI → power, no learning) | 3.29% |

Averaged over a 48-origin rolling walk-forward backtest across a full
year: model 1.34% vs. smart persistence 2.12% vs. persistence 2.10%.
Full numbers, methodology, and the exact data window/seeds used are in
`src/models/model_card.json`.

**What this number is not:** a claim about real-world accuracy. It is
scored entirely on `src/ingestion/jaipur_simulator.py`'s synthetic data —
90 days of one location, one simulated plant, procedurally generated
weather. No seasonal, geographic, or real-plant claim is defensible from
this yet. The model previously reported here (an unrelated, now-removed
2.17% MAPE figure) suffered from target leakage and is not comparable to
the number above — see `git log` for the fix.

---

## Architecture

```
Open-Meteo (live weather, dashboard) ──┐
                                        ↓
              src/features/pipeline.py (shared: training, API, dashboard)
                                        ↓
              XGBoost model (src/models/xgboost_solar_v2.json)
                                        ↓
         ┌──────────────────────────────────────────┐
         ↓                                            ↓
  Standalone Streamlit dashboard              FastAPI service (src/api/)
  (rule-based battery optimizer inline)       (not part of the live deploy)
```

The deployed dashboard (`dashboard/app.py`) runs standalone — it loads the
model and calls Open-Meteo directly, with no FastAPI hop. `docker compose
up` runs both the dashboard and the API together (see `docker-compose.yml`);
a single `docker run` gets the standalone dashboard only.

---

## Project Structure

```
.
├── configs/            # All configuration — nothing hardcoded in code
├── src/
│   ├── ingestion/      # Open-Meteo fetcher, Jaipur data simulator, synthetic forecast-error model
│   ├── features/       # pipeline.py — the one feature implementation (training, API, dashboard all import it)
│   ├── models/         # train.py, model_card.py, the served model + its model card
│   ├── optimization/   # PuLP-based battery dispatch optimizer (not yet wired into the live dashboard)
│   └── api/            # FastAPI service (standalone — not used by the deployed dashboard)
├── dashboard/          # Standalone Streamlit app — this is what's deployed
├── notebooks/          # Archived exploration: EDA, LSTM/Prophet/ensemble experiments (not served — see Known Gaps)
├── benchmark.py         # Honest evaluation: rolling-origin backtest + baselines
├── Makefile             # make train / make benchmark / make test
└── tests/               # Unit tests, run in CI on every push (.github/workflows/)
```

---

## Quickstart

```bash
# Clone and set up
git clone https://github.com/Siddarth070/Solar-Forecasting-And-Optimization-AI-Platform.git
cd Solar-Forecasting-And-Optimization-AI-Platform

# Install dependencies (fully pinned)
pip install -r requirements.txt

# Rebuild the model from scratch (synthetic data, deterministic given
# the seeds in src/models/model_card.json)
make train

# Score it honestly against baselines
make benchmark

# Run the dashboard locally
streamlit run dashboard/app.py

# Or run both the dashboard and the API together
docker compose up --build

# Run tests
make test
```

---

## Status

Being rebuilt phase by phase against an internal execution roadmap, gated
on a Phase 0 credibility checklist (target leakage, train/serve parity,
honest evaluation, reproducibility, dependency hygiene, this README) before
any new feature work. As of this commit:

- **Done:** target leakage removed and tested in CI; one shared feature
  pipeline (no more hand-duplicated feature dicts in the API/dashboard);
  a feature/serving parity test; the honest evaluation harness above;
  model-skill vs. deliverable-skill split; reproducible training
  (`make train`, JSON model format, `model_card.json`); Docker build
  fixed (non-root, both services start, verified in CI); dependencies
  fully pinned.
- **Not started:** per-plant/multi-state configuration, a proper 15-minute
  time-block engine, probabilistic (P10/P50/P90) forecasts, real
  customer-file ingestion, loss attribution, and any real-plant
  validation. All of Phases 1 onward.

## Known Gaps

- The deployed dashboard's battery optimizer is a simple rule-based
  heuristic, not the PuLP linear program in `src/optimization/` — and
  that LP itself currently has no round-trip efficiency and only supports
  hourly (not 15-minute) blocks.
- LSTM and Prophet were explored in `notebooks/` but are not served; only
  XGBoost is in the product path. Those notebooks' own reported metrics
  predate the target-leakage fix and are marked invalid in-notebook.
- No live weather-forecast-error data exists yet — the "deliverable
  skill" number above uses a clearly-labeled synthetic proxy for D-1
  forecast error (`src/ingestion/synthetic_forecast.py`), not a real
  forecast archive.
- Everything is trained and validated on simulated data for one plant in
  one location — no real-plant or multi-season (beyond a synthetic full
  year) validation exists yet.

---

## Tech Stack

**Data:** Python, Pandas, NumPy, PyArrow
**Forecasting:** XGBoost, pvlib (solar-position clear-sky modeling)
**Optimization:** PuLP
**API:** FastAPI
**Dashboard:** Streamlit
**Infrastructure:** Docker, Docker Compose, GitHub Actions CI

---

## Target Users

The wedge is small-to-mid operators who feel forecast error directly,
not utility-scale grid entities (procurement and integration cycles there
are too long to survive on pre-revenue cash — that's deliberately a much
later step):

- **Solar IPPs (5–100 MW)** — scheduling accuracy to reduce DSM/deviation penalties
- **C&I plant owners** — visibility into their own generation without building it themselves
- **Regional O&M companies** — automate schedule-vs-actual reporting they currently do by hand
- **QCA / scheduling consultants** — reduce manual 96-block schedule preparation
- **Battery storage operators** — charge/discharge scheduling (once the optimizer above is fixed — see Known Gaps)

---

## Data Sources

- [Open-Meteo](https://open-meteo.com) — free hourly weather forecasts, used live by the dashboard. No API key required.
- `src/ingestion/jaipur_simulator.py` — the actual source of training data today: a seeded, deterministic synthetic generator (see Honest Results above for its limitations).

---

*Built as a portfolio project demonstrating end-to-end, honestly-evaluated
ML engineering for the Indian energy sector. Target location for the
current simulation: Jaipur, Rajasthan.*

## Live Demo
👉 [Open Dashboard](https://zenith-to.streamlit.app/) — trained entirely on
synthetic data (see Honest Results above); not yet validated against a
real plant.
