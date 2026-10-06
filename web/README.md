# Zenith web console

React + TypeScript (Vite) front end for the FastAPI service in `src/api/`.
Every number on screen comes from a real API call; nothing is mocked.

| Page | What it does | API calls |
|---|---|---|
| `/` Overview | Positioning, honest build status, pilot ask; hero strip is tomorrow's real forecast | `/plants/{id}`, `/forecast` |
| `/console` | Forecast → choose a declaration → DSM loss for 3 declarations × 3 outcomes → battery dispatch → intraday revision; CSV export of the 96-block schedule | `/plants`, `/plants/{id}`, `/forecast`, `/optimize` (×9 + battery), `/schedule/gate-closures`, `/schedule/revise` |
| `/model` | Walk-forward backtest and holdout vs. baselines, from the model card | `/model-card` |

Weather comes from Open-Meteo in the browser (no key). See `src/lib/weather.ts`
for the three alignment rules: wind in m/s, radiation stamped t+15 belongs to
the block starting at t, and IST local times. A clear-sky demo mode exists for
when Open-Meteo is unreachable and is labelled everywhere it's used.

**DSM loss** (`src/lib/blocks.ts`) = settlement + signed deviation × contract
rate, i.e. what DSM costs versus having declared exactly what was delivered.
It is zero inside the first volume band. Verified equal to
`rate × (0.1 × band-2 MWh + band-3 MWh)` from `src/regulatory/dsm.py`.

## Run

```bash
# API (from repo root)
uvicorn src.api.main:app --port 8000

# Console, with /api proxied to :8000
cd web && npm install && npm run dev      # http://localhost:5173
npm test                                  # unit tests (weather alignment, DSM loss, IST)
```

Or everything at once from the repo root: `docker compose up --build`, then
open http://localhost:8080 (nginx serves the build and proxies `/api`).

## Deploy

- **Static site (Netlify, zenith-energy.in):** base directory `web`; `netlify.toml` handles the build and SPA routes.
- **API:** host the existing Docker image anywhere that runs a container (Render, Railway, Fly, a VM) with `uvicorn src.api.main:app --host 0.0.0.0 --port 8000`.
- Set `VITE_API_URL=https://<your-api-host>` in Netlify, and `ZENITH_CORS_ORIGINS=https://zenith-energy.in,https://www.zenith-energy.in` on the API.
