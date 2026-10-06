import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { BatteryPanel } from "../components/BatteryPanel";
import { DeclarationPanel, type LossMatrix } from "../components/DeclarationPanel";
import { ForecastPanel } from "../components/ForecastPanel";
import { RevisionPanel } from "../components/RevisionPanel";
import { NO_BATTERY, api } from "../lib/api";
import { STRATEGIES, addDays, declaredFor, istToday, outcomeSeries, summarizeLoss, type Outcome, type Strategy } from "../lib/blocks";
import { longDate } from "../lib/format";
import { useDayForecast, type WeatherSource } from "../lib/useDayForecast";

const OUTCOMES: Outcome[] = ["p10", "p50", "p90"];

export function Console() {
  const [params, setParams] = useSearchParams();
  const today = istToday();
  const tomorrow = addDays(today, 1);

  const date = params.get("date") ?? tomorrow;
  const source = (params.get("weather") as WeatherSource) ?? "open-meteo";
  const strategy = (params.get("declare") as Strategy) ?? "p50";
  const [plantIds, setPlantIds] = useState<string[] | null>(null);
  const [plantsError, setPlantsError] = useState<string | null>(null);
  const plantId = params.get("plant") ?? plantIds?.[0] ?? null;
  const [reload, setReload] = useState(0);

  const update = (patch: Record<string, string>) => {
    const next = new URLSearchParams(params);
    Object.entries(patch).forEach(([k, v]) => next.set(k, v));
    setParams(next, { replace: true });
  };

  useEffect(() => {
    api.plants().then((r) => setPlantIds(r.plant_ids)).catch((e) => setPlantsError(e.message));
  }, [reload]);

  const s = useDayForecast(plantId, date, source, reload);
  const { forecast, plant, weather } = s;
  const declared = useMemo(() => (forecast ? declaredFor(forecast, strategy) : []), [forecast, strategy]);

  // Price every declaration against every outcome with no battery: the
  // baseline DSM exposure. Nine /optimize calls, run together.
  const [matrix, setMatrix] = useState<LossMatrix | null>(null);
  const [matrixError, setMatrixError] = useState<string | null>(null);
  useEffect(() => {
    setMatrix(null);
    setMatrixError(null);
    if (!forecast || !plant) return;
    const rate = plant.regulatory.contract_rate_rs_per_kwh;
    if (rate == null) return;
    let cancelled = false;
    const jobs = STRATEGIES.flatMap((st) =>
      OUTCOMES.map(async (o) => {
        const res = await api.optimize({
          plantId: plant.plant_id,
          date,
          solarMw: outcomeSeries(forecast, o),
          declaredMw: declaredFor(forecast, st.id),
          battery: NO_BATTERY,
        });
        return [st.id, o, summarizeLoss(res.schedule, rate)] as const;
      }),
    );
    Promise.all(jobs)
      .then((cells) => {
        if (cancelled) return;
        const m = {} as LossMatrix;
        for (const [st, o, sum] of cells) (m[st] ??= {} as LossMatrix[Strategy])[o] = sum;
        setMatrix(m);
      })
      .catch((e) => !cancelled && setMatrixError(e.message));
    return () => {
      cancelled = true;
    };
  }, [forecast, plant, date]);

  const datePreset = date === today ? "today" : date === tomorrow ? "tomorrow" : "other";

  return (
    <main className="console">
      <div className="wrap">
        <div className="console-head">
          <div>
            <h1 style={{ fontSize: "var(--t-xl)" }}>Scheduling console</h1>
            <p className="muted" style={{ marginTop: 8 }}>
              {plant ? `${plant.name}, ${plant.location.name}, ${plant.location.state}, ${plant.ac_capacity_mw} MW AC. ` : ""}
              {longDate(date)}, 96 blocks IST.
            </p>
          </div>
          <div className="controls">
            <label className="field">
              Plant
              <select value={plantId ?? ""} onChange={(e) => update({ plant: e.target.value })} disabled={!plantIds}>
                {(plantIds ?? []).map((id) => (
                  <option key={id} value={id}>{id}</option>
                ))}
              </select>
            </label>
            <div className="field">
              <span>Day</span>
              <div className="segmented" role="group" aria-label="Day">
                <button aria-pressed={datePreset === "today"} onClick={() => update({ date: today })}>Today</button>
                <button aria-pressed={datePreset === "tomorrow"} onClick={() => update({ date: tomorrow })}>Tomorrow</button>
              </div>
            </div>
            <label className="field">
              Or pick a date
              <input
                type="date"
                value={date}
                min={addDays(today, -60)}
                max={addDays(today, 14)}
                onChange={(e) => e.target.value && update({ date: e.target.value })}
              />
            </label>
            <div className="field">
              <span>Weather</span>
              <div className="segmented" role="group" aria-label="Weather source">
                <button aria-pressed={source === "open-meteo"} onClick={() => update({ weather: "open-meteo" })}>Open-Meteo</button>
                <button aria-pressed={source === "demo"} onClick={() => update({ weather: "demo" })}>Clear-sky demo</button>
              </div>
            </div>
          </div>
        </div>

        {plant?.simulated && (
          <p className="notice">
            This is a simulated plant and the model has only been trained on simulated data. Use these numbers to
            evaluate the workflow, not to submit a real schedule.
          </p>
        )}

        {plantsError && (
          <div className="panel">
            <p className="notice error">{plantsError}</p>
            <div className="row-actions">
              <button className="btn" onClick={() => { setPlantsError(null); setReload((r) => r + 1); }}>Try again</button>
            </div>
          </div>
        )}

        {s.status === "loading" && <p className="loading">{s.step}…</p>}

        {s.status === "error" && (
          <div className="panel">
            <p className="notice error">{s.error}</p>
            <div className="row-actions">
              <button className="btn" onClick={() => setReload((r) => r + 1)}>Try again</button>
              {source === "open-meteo" && (
                <button className="btn secondary" onClick={() => update({ weather: "demo" })}>Use clear-sky demo weather</button>
              )}
            </div>
          </div>
        )}

        {s.status === "ready" && forecast && plant && weather && (
          <>
            <ForecastPanel forecast={forecast} declared={declared} plant={plant} weather={weather} />
            <DeclarationPanel
              plant={plant}
              date={date}
              forecast={forecast}
              declared={declared}
              strategy={strategy}
              onStrategy={(st) => update({ declare: st })}
              matrix={matrix}
              matrixError={matrixError}
            />
            <BatteryPanel
              key={`${plant.plant_id}-${date}-${strategy}-${source}`}
              plant={plant}
              date={date}
              forecast={forecast}
              declared={declared}
              lossWithout={matrix ? matrix[strategy] : null}
            />
            <RevisionPanel key={`${plant.plant_id}-${date}`} plant={plant} date={date} forecast={forecast} declared={declared} strategy={strategy} />
          </>
        )}
      </div>
    </main>
  );
}
