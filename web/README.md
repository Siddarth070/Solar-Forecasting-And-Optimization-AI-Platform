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

## Deploy (make it public)

Two pieces: the static site (Netlify) and the API (Render). The console
only ever *reads* from the API, so the API runs read-only in public.

1. **API on Render.** New -> Blueprint -> this repo; `render.yaml` at the
   repo root defines the service. When prompted, set
   `ZENITH_CORS_ORIGINS` to the site's exact origins (step 3's domain plus
   the `https://<name>.netlify.app` URL). Check
   `https://<api>.onrender.com/health` returns `"model_loaded": true`.
2. **Site on Netlify.** Add new site -> import this repo -> base directory
   `web` (`netlify.toml` does the build and SPA routes). Under Site
   configuration -> Environment variables set
   `VITE_API_URL=https://<api>.onrender.com`, then redeploy (Vite bakes it
   in at build time).
3. **Domain.** Buy it, add it under Netlify -> Domain management (apex +
   `www`), point DNS as Netlify shows; HTTPS is automatic. Optionally add
   `api.<domain>` as a Render custom domain and use that in
   `VITE_API_URL`. Add every origin you serve the site from to
   `ZENITH_CORS_ORIGINS`.
4. **Check.** Open the site; the browser console must show no CORS
   errors; `curl -X POST https://<api>/plants` must return 403.

Public-API guards (`src/api/main.py`), all configured by env vars:

| Variable | Default | Effect |
|---|---|---|
| `ZENITH_CORS_ORIGINS` | local dev origins | Browser origins allowed to call the API. Never `*`. |
| `ZENITH_WRITE_API_KEY` | unset | If set, write endpoints need `X-API-Key: <value>`. |
| `ZENITH_ALLOW_OPEN_WRITES` | unset | `1` opens write endpoints with no key. Local/docker compose only. |
| `ZENITH_RATE_LIMIT_PER_MIN` | `60` | POSTs per client per minute (`0` = off). Bodies over 1 MB get 413. |
| `ZENITH_TRUST_PROXY` | unset | `1` behind Render/Railway/Fly so clients are told apart by IP. |

Known limits of the free tiers: Render's free plan sleeps when idle (the
first request after that takes ~30-60 s) and its disk is wiped on every
deploy, so anything written through the API (onboarded plants,
recommendation decisions) does not persist there.
