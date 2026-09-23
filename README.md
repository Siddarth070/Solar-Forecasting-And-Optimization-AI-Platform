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
├── configs/
│   ├── config.yaml     # Global, plant-agnostic settings only
│   └── plants/         # One YAML per plant (location, capacity) — see src/utils/config_loader.get_plant_config
├── src/
│   ├── ingestion/      # Open-Meteo fetcher, Jaipur data simulator, synthetic forecast-error model
│   ├── features/       # pipeline.py — the one feature implementation (training, API, dashboard all import it)
│   ├── models/         # train.py, model_card.py, the served model + its model card
│   ├── optimization/   # PuLP-based battery dispatch optimizer (not yet wired into the live dashboard)
│   ├── regulatory/     # dsm.py (DSM charges) + grid_code.py (revision/gate-closure timing) — real CERC rules
│   ├── scheduling/     # rolling_horizon.py — applies a schedule revision under grid_code.py's gate-closure timing
│   └── api/            # FastAPI service (standalone — not used by the deployed dashboard)
├── dashboard/          # Standalone Streamlit app — this is what's deployed
├── notebooks/          # Archived exploration — see notebooks/README.md (not served, roadmap P1.8)
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

- **Phase 0, done:** target leakage removed and tested in CI; one shared
  feature pipeline (no more hand-duplicated feature dicts in the
  API/dashboard); a feature/serving parity test; the honest evaluation
  harness above; model-skill vs. deliverable-skill split; reproducible
  training (`make train`, JSON model format, `model_card.json`); Docker
  build fixed (non-root, both services start, verified in CI);
  dependencies fully pinned.
- **Phase 1, partially done:** per-plant configuration (`configs/plants/`
  — two plants in two states run from the same binary, tested in CI; the
  model trains on capacity FRACTION so one artifact correctly serves
  differently-sized plants); the dashboard's fake sine-wave "demand"
  curve replaced with an operator-set declared schedule; a resolution-
  independent 96-block/15-minute time engine (`src/time_blocks.py`,
  matching India's real grid-scheduling grid); the battery optimizer
  rewritten as a proper LP with round-trip efficiency, SOC floor/ceiling,
  a terminal SOC constraint, and an objective that minimizes DSM
  exposure (deviation from the declared schedule) in both directions,
  not "unmet demand" only; probabilistic P10/P50/P90 forecasts (a second
  XGBoost model with a multi-output quantile objective, served alongside
  the point forecast in `/forecast` and shown as an uncertainty band on
  the dashboard), validated with pinball loss and a reliability check on
  the final holdout (see `src/models/model_card.json`'s
  `quantile_final_holdout`); a config-driven DSM ruleset
  (`src/regulatory/dsm.py`) implementing the real CERC (Deviation
  Settlement Mechanism and Related Matters) Regulations, 2024,
  Regulation 8(4) tiered WS-seller charge structure (as amended — the
  amendments read don't change this structure; see the module's own
  docstring for full citations and the one real gap it documents rather
  than fabricates: CERC's post-01.04.2026 deviation-% blend weight is not
  yet published), wired into the battery LP's objective (`POST
  /optimize?plant_id=...`) and reported (not yet optimized-for) on the
  dashboard; rolling day-ahead/intraday horizons (`src/regulatory/
  grid_code.py`, `src/scheduling/rolling_horizon.py`) implementing the
  real CERC Indian Electricity Grid Code (IEGC) 2023 Regulation 49(4)(c)
  gate-closure rule — a requested schedule revision only takes effect 6-7
  full 15-minute blocks later, verified against a CERC removal-of-
  difficulties order's own worked numerical example — plus Regulation
  49(8)'s restriction that a WS seller may only revise under a bilateral
  transaction structure, not a collective one; plus the Real-Time Market
  (RTM) gate-closure mechanism (Regulation 49(1)(q)) — half-hour delivery
  windows whose bid window opens 75 minutes and closes 60 minutes before
  each window starts, verified against the regulation's own concrete
  worked instance (22:45–23:00 hrs bidding for the 00:00–00:30 delivery
  window). Both served via `POST /schedule/revise` and `GET
  /schedule/gate-closures` on the real 96-block grid; `POST /forecast`
  now takes a real per-reading ISO timestamp instead of the old
  hour+month pair anchored to a fixed placeholder date, and `POST
  /optimize` enforces exact 96-block/15-minute alignment (and forces
  `dt_hours=0.25` to match) whenever an optional `date` is supplied,
  returning real block-start timestamps — the same alignment
  `/schedule/revise` already used. **Not done:** a numeric revision-count
  cap for WS sellers specifically (none of the source documents specify
  one; see `grid_code.py`'s docstring).
- **Phase 2, started:** a data-quality gate (`src/quality/gate.py`,
  `POST /quality/check`) that runs before any forecast is shown — per
  the roadmap's own framing, "the customer sees a quality report before
  they see a forecast." It checks ingested generation readings for a
  missing timezone, duplicate timestamps, timestamp gaps (nominal
  resolution inferred as the mode of consecutive deltas, robust to a few
  genuine gaps), negative power, above-rated-capacity readings, flatlines
  (a non-zero value repeated for >= 4 consecutive blocks — a likely stuck
  inverter, distinct from the normal run of identical zero readings every
  night), and generation reported while the sun is below the horizon at
  the plant's real location (`pvlib`'s clear-sky model, using each
  plant's actual lat/lon from `configs/plants/`). Each check reports a
  severity (error = unsafe to forecast on, warning = usable but noted),
  a count, and the specific timestamps — never a bare pass/fail. Tested
  with 14 unit tests (one engineered problem per test, checked not to
  accidentally trip a second check) plus 5 end-to-end API tests.
  A loss attribution engine (`src/attribution/loss.py`, `POST
  /losses/attribute`) that compares actual generation against
  expected-from-irradiance — reusing the exact PV physics formula
  `src/ingestion/jaipur_simulator.py` uses to generate this project's own
  training data (GHI × temperature-derated performance ratio × capacity),
  just parameterized by each plant's own config instead of that
  simulator's hardcoded constants — and classifies each block's material
  residual loss into a cause with the evidence that produced it: weather
  (a clear-sky-index ramp consistent with a passing cloud), equipment
  (a sustained, irradiance-uncorrelated drop that still tracks the
  underlying solar ramp's shape), curtailment (a hard flat output clip
  while expected output keeps varying), or unknown (material loss with
  no supporting pattern). A separate day-level check
  (`daily_performance_ratio()` / `detect_soiling()`) flags a slow,
  sustained decline in the daily actual/expected ratio over 10+ days as
  suspected soiling — a trend invisible at single-block granularity.
  Every classification carries a confidence label and the raw numbers
  behind it, never a bare label. Tested with 12 unit tests (one
  hand-computed loss scenario per cause, with exact expected MW values
  worked out by hand, not just "it ran") plus 6 end-to-end API tests.
  **Honestly limited, not fabricated:** this platform has no
  per-inverter/string telemetry and no real grid curtailment-instruction
  feed, so "equipment" and "curtailment" are always reported as
  suspected/possible — the module's own docstring and every such block's
  evidence text say so explicitly, never claiming a confirmed asset or a
  confirmed grid order.
  A schedule-risk scorer (`src/risk/schedule_risk.py`, `POST
  /schedule/risk`) that runs each block's P10/P50/P90 forecast through
  the real CERC DSM settlement math (`src/regulatory/dsm.py`, roadmap
  P1.7) against the declared schedule and reports which Note-1
  volume-limit band each quantile's deviation reaches — reusing
  `deviation_settlement()`'s own segment widths to determine the band
  rather than re-deriving separate cutoffs, so the band can never
  disagree with the Rs figure reported alongside it. A block's risk
  level (low/medium/high) is driven by the WORST band any of the three
  quantiles reaches, so a calm median with a wide, risky tail is still
  flagged — not just the point forecast. Correctly reads the real
  01.04.2026 DSM cutover date (`as_of`), tightening the bands
  post-cutover exactly as Regulation 8(4) Note-1 specifies. Per the
  roadmap, every block AND the report itself carry the disclaimer
  verbatim: "Indicative DSM exposure based on configured assumptions and
  uploaded data. Not an official settlement statement." Has no flat-
  penalty fallback (unlike `/optimize`) — a plant missing
  `regulatory.seller_category`/`contract_rate_rs_per_kwh` gets a clear
  422, not a silent, meaningless estimate. Tested with 7 unit tests
  (hand-computed deviation MWh and band per scenario, including a test
  proving the same deviation reads a stricter band after the cutover
  date) plus 7 end-to-end API tests.
  An operator-recommendation engine (`src/recommendations/engine.py` +
  `store.py`, `POST /recommendations/generate`, `GET /recommendations`,
  `POST /recommendations/{id}/decide`) that turns the already-computed
  P2.4/P2.5 evidence into concrete suggestions — per the roadmap,
  **human-approved only, never automatic control**: nothing in this
  engine writes to a schedule, a battery, or anything else by itself.
  Three rules: a schedule-revision suggestion for each HIGH-risk block
  whose real Grid Code revision gate (`src/regulatory/grid_code.py`,
  roadmap P1.4) is still open — skipped outright, not just gated, for a
  plant whose configured transaction structure isn't revision-eligible
  at all (Regulation 49(8)); a battery charge/discharge suggestion for
  each MEDIUM/HIGH-risk block the plant's actual battery headroom can
  (even partially) offset, explicitly marked "partial" when it can't
  fully cover the deviation; and one inspection suggestion per
  suspected-equipment RUN (grouped by that run's own start timestamp, so
  one physical fault produces one suggestion, not one per 15-minute
  block). Every suggestion carries its trigger and the raw evidence that
  produced it. Approve/dismiss decisions are logged to an append-only
  SQLite log (`src/recommendations/store.py`) — chosen because this
  platform has no real database yet (that's roadmap P2.11, a separate
  architecture decision) and an in-memory log would lose the audit trail
  on every restart; a later decision can override an earlier one's
  status without erasing it from the log. Tested with 13 unit tests for
  the three rules plus 10 for the log (including that a nonexistent
  recommendation, an unattributed decision, and an invalid decision
  value all fail loudly) plus 10 end-to-end API tests.
  A weekly forecast-performance report (`src/reporting/weekly_report.py`,
  `POST /reports/weekly`) that scores forecasts that were ACTUALLY
  SERVED, once the real outcome is known, by lead time (horizon) and time
  of day (block) — exactly where a forecast tends to be weakest — against
  the same three untrained baselines (persistence, smart_persistence,
  physics) `benchmark.py` already used for training-time evaluation.
  Those baselines were factored out into `src/evaluation/baselines.py`
  first, so `benchmark.py` and this new production report use identical
  math, not two implementations that could quietly drift apart. Reports
  a `beats_baseline` verdict per baseline (model nMAE below that
  baseline's, computed, never assumed) alongside the raw numbers. Per the
  roadmap, "contains no number that cannot be traced to raw data" —
  every figure is a direct nMAE/nRMSE off the actual/predicted pairs
  given; nothing is estimated. Rejects a request whose forecasts aren't
  fully covered by the supplied readings (a report cannot honestly score
  an outcome it wasn't given) with a clear 422. Tested with 9 unit tests
  for the shared baselines (`tests/test_baselines.py`) plus 7 for the
  report itself — every nMAE hand-computed from a small, fully
  deterministic dataset (a constant clear-sky index so the physics
  baseline is a known constant, and two different constant-actual days so
  persistence/smart_persistence carry an exact, known error) — plus 5
  end-to-end API tests.
- **Not started:** real customer-file ingestion (P2.1) and any
  real-plant validation — rest of Phase 2 onward.

## Known Gaps

- The weather SIMULATOR (`src/ingestion/jaipur_simulator.py`) is still
  Jaipur-specific — `configs/plants/pune_50mw.yaml` proves the
  config/serving layer is plant-agnostic, not that a real Pune-trained
  model exists. That needs a location-aware simulator or real per-plant
  data (Phase 3).
- The deployed dashboard's battery optimizer is still a simple rule-based
  heuristic, not the PuLP linear program in `src/optimization/` (which now
  has round-trip efficiency, SOC floor/ceiling, a terminal SOC constraint,
  and the real CERC DSM charge structure) — wiring the dashboard to call
  the real LP is not done yet; it only *reports* an estimated DSM cost for
  its own heuristic's dispatch, using `src/regulatory/dsm.py`.
- `contract_rate_rs_per_kwh` in `configs/plants/*.yaml` is an illustrative
  placeholder (Rs 2.50/kWh), not a real PPA/auction tariff — these are
  simulated demo plants with no real commercial contract to cite. The DSM
  *rate structure* itself (`src/regulatory/dsm.py`) is real and verified
  against the CERC regulation text; only this one commercial input needs
  a real number before the Rs figures mean anything for an actual plant.
- CERC's post-01.04.2026 deviation-% blend weight ("X" in Regulation
  6(2)(b)) has not yet been published by separate order — `dsm.py`
  documents this and uses the one fully-specified fallback (Available
  Capacity alone) rather than inventing a value.
- The probabilistic (P10/P50/P90) model is trained on the same hourly
  simulator rows as the point model, and `POST /forecast` still returns
  one prediction per requested (hourly) timestamp, not per 15-minute
  block — there is no automatic upsampling from an hourly forecast onto
  the 96-block grid. `POST /optimize` and `POST /schedule/revise` DO
  enforce that grid on their own inputs, but a caller chaining
  `/forecast` into either of them today must resample the 12-24 hourly
  values onto 96 blocks itself (e.g. via `src.time_blocks.
  integrate_to_blocks`, the same tool P1.2 built for exactly this).
- `POST /schedule/revise`'s gate-closure timing (`src/regulatory/
  grid_code.py`) is real and verified against a CERC order's own worked
  example, but nothing in this platform yet SUBMITS a real day-ahead
  schedule to any real Load Despatch Centre — it only enforces the
  timing rule on schedules the caller supplies. `transaction_type:
  "bilateral"` in `configs/plants/*.yaml` is an assumption (a solar IPP
  with a PPA is typically bilateral), not a verified fact about a real
  plant's actual sale structure.
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
- The data-quality gate (`src/quality/gate.py`) cannot detect a
  mislabeled timezone by inference (e.g. data secretly in UTC but claiming
  to be IST) — that needs a real ground truth to compare against. Its
  `night_time_non_zero` check catches the common ~5:30h UTC/IST offset
  error as a side effect (real solar position at the claimed timestamp
  would show the sun down while data reports generation), but a
  wrongly-labeled timezone with a smaller offset would not necessarily
  trip it. It has also only been exercised on synthetic data — no real
  customer file, with real logger dropouts and re-export duplicates, has
  been run through it yet.
- The loss attribution engine (`src/attribution/loss.py`) works from one
  plant-level aggregate power reading and a single site-wide GHI/
  temperature pair — it has no per-inverter or per-string telemetry, so
  it can never confirm WHICH asset caused an "equipment" loss, only that
  the aggregate pattern looks like one. It also has no real grid
  curtailment-instruction feed (SLDC/RLDC order log), so a hard flat clip
  is reported as "possible curtailment", never a confirmed grid-ordered
  one — that confirmation needs a real QCA/grid-operator integration
  (roadmap P2.7/P2.8, both explicitly deferred pending a real
  counterparty). The soiling check needs 10+ real calendar days of
  readings to fit a trend on; it has only been exercised on synthetic
  multi-day scenarios, not a real degrading plant.
- The schedule-risk scorer (`src/risk/schedule_risk.py`) inherits, not
  duplicates, `src/regulatory/dsm.py`'s own two documented gaps: its Rs
  figures use the illustrative `contract_rate_rs_per_kwh` placeholder
  (not a real PPA tariff), and its post-01.04.2026 classification uses
  the same explicitly-labeled Available-Capacity-only fallback for the
  unpublished blend-weight "X". It also has no automatic upsampling from
  `/forecast`'s hourly P10/P50/P90 output onto the 96-block grid this
  endpoint expects (the same gap already noted for `/forecast` above) —
  a caller must resample first, e.g. via `src.time_blocks.
  integrate_to_blocks`.
- The operator-recommendation log (`src/recommendations/store.py`) is a
  single SQLite file on the API server's own disk, not a shared,
  multi-instance-safe database — fine for one server process (this
  platform's current deployment shape), but it would need a real
  database (roadmap P2.11) before running behind more than one API
  instance. The recommendation rules themselves only see what's already
  in a P2.4/P2.5 report, so they inherit every gap already documented
  for those: no per-inverter telemetry behind an "inspection"
  suggestion, no real grid curtailment feed, and the illustrative
  contract-rate placeholder behind a schedule-revision suggestion's Rs
  figures. Nothing here has been exercised against a real operator's
  actual workflow — only synthetic trigger scenarios.
- `POST /reports/weekly` (roadmap P2.9) has no forecast log to draw
  on — `POST /forecast` doesn't persist what it returns anywhere, so a
  caller must supply the served forecasts and their now-known actual
  outcomes itself. This module has only been exercised on synthetic
  data; wiring a real forecast log (so this report could genuinely run
  unattended on a weekly cadence against live history) is future work,
  as is the scheduler/cron layer itself — "generated unattended" here
  describes the scoring math (deterministic, no human judgment calls),
  not an actual deployed schedule.

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
