import { useEffect, useState } from "react";
import { CartesianGrid, ComposedChart, Legend, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, type Forecast, type GateClosures, type Plant, type RevisionResult } from "../lib/api";
import { STRATEGIES, blockClock, declaredFor, istClock, istNowIso, istToday, type Strategy } from "../lib/blocks";
import { TICKS } from "./ForecastPanel";

export function RevisionPanel(props: { plant: Plant; date: string; forecast: Forecast; declared: number[]; strategy: Strategy }) {
  const { plant, date, forecast, declared, strategy } = props;
  const [gates, setGates] = useState<GateClosures | null>(null);
  const [gateError, setGateError] = useState<string | null>(null);
  const [proposed, setProposed] = useState<Strategy>(strategy === "p50" ? "p10" : "p50");
  const isToday = date === istToday();
  const [time, setTime] = useState(() => (isToday ? istNowIso().slice(11, 16) : "09:00"));
  const [result, setResult] = useState<RevisionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    setGateError(null);
    api.gateClosures(plant.plant_id, istNowIso()).then(setGates).catch((e) => setGateError(e.message));
  }, [plant.plant_id]);

  useEffect(() => setResult(null), [date, strategy, proposed, time, plant.plant_id]);

  // Keep the revision target different from what's already declared.
  useEffect(() => {
    if (proposed === strategy) setProposed(strategy === "p50" ? "p10" : "p50");
  }, [strategy]); // eslint-disable-line react-hooks/exhaustive-deps

  async function run() {
    setBusy(true);
    setError(null);
    try {
      setResult(
        await api.revise({
          plantId: plant.plant_id,
          date,
          locked: declared,
          proposed: declaredFor(forecast, proposed),
          requestTimestamp: `${date}T${time}:00+05:30`,
        }),
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  const proposedSeries = declaredFor(forecast, proposed);
  const data = declared.map((d, i) => ({ i, locked: d, proposed: proposedSeries[i], applied: result?.applied_schedule_mw[i] }));
  const sameAsDeclared = proposed === strategy;

  return (
    <section className="panel" aria-labelledby="rev-h">
      <header>
        <div>
          <h2 id="rev-h">Intraday revision</h2>
          <p>
            A revision doesn't apply from the moment you ask. Under IEGC 2023 Reg. 49 it takes effect several blocks
            later, and only for bilateral sellers. Check what would actually change.
          </p>
        </div>
      </header>

      <div className="split">
        <div>
          <h3 style={{ fontSize: "1rem", marginBottom: 10 }}>Right now</h3>
          {gateError && <p className="notice error">{gateError}</p>}
          {gates && (
            <dl className="kv">
              <dt>Sells as</dt>
              <dd>{gates.transaction_type === "bilateral" ? "Bilateral (can revise)" : "Collective (can't revise)"}</dd>
              <dt>A revision now applies from</dt>
              <dd>{gates.bilateral_revision.effective_timestamp ? istClock(gates.bilateral_revision.effective_timestamp) : "Not allowed"}</dd>
              <dt>Real-time market window</dt>
              <dd>Delivery from {istClock(gates.real_time_market.delivery_window_start)}, bids close {istClock(gates.real_time_market.gate_closure)}</dd>
            </dl>
          )}
          {plant.regulatory.transaction_type === "bilateral" && plant.simulated && (
            <p className="small muted" style={{ marginTop: 10 }}>Bilateral is assumed for this simulated plant.</p>
          )}

          <h3 style={{ fontSize: "1rem", margin: "22px 0 10px" }}>Try a revision</h3>
          <div className="param-grid">
            <label className="field">
              Revise to
              <select value={proposed} onChange={(e) => setProposed(e.target.value as Strategy)}>
                {STRATEGIES.map((s) => (
                  <option key={s.id} value={s.id}>{s.short}</option>
                ))}
              </select>
            </label>
            <label className="field">
              Requested at (IST)
              <input type="time" value={time} onChange={(e) => setTime(e.target.value)} />
            </label>
          </div>
          {sameAsDeclared && <p className="small muted" style={{ marginTop: 8 }}>That's what you already declared, so nothing would change.</p>}
          <div className="row-actions">
            <button className="btn" onClick={run} disabled={busy || sameAsDeclared}>
              {busy ? "Checking…" : "Check revision"}
            </button>
          </div>
        </div>

        <div>
          {error && <p className="notice error">{error}</p>}
          {result && (
            <p className={result.revision_allowed ? "notice info" : "notice error"} style={{ marginBottom: 12 }}>
              {result.revision_allowed && result.effective_timestamp
                ? `Requested at ${time}, the revision applies from ${istClock(result.effective_timestamp)}. ${result.locked_block_count} of 96 blocks stay at the original declaration.`
                : "This plant sells collectively, so Reg. 49(8) doesn't allow it to revise. The original schedule stands."}
            </p>
          )}
          <div className="chart-box" style={{ height: 260 }} role="img" aria-label="Declared, proposed and applied schedules">
            <ResponsiveContainer>
              <ComposedChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -12 }}>
                <CartesianGrid stroke="#e6ebf0" vertical={false} />
                <XAxis dataKey="i" ticks={TICKS} tickFormatter={(i: number) => (i === 95 ? "24:00" : blockClock(i))} tick={{ fontSize: 12, fill: "#6a7a8f" }} />
                <YAxis domain={[0, plant.ac_capacity_mw]} tick={{ fontSize: 12, fill: "#6a7a8f" }} width={60} />
                <Tooltip labelFormatter={(i: number) => `Block ${i + 1}, ${blockClock(i)}`} formatter={(v: number, n: string) => [`${v.toFixed(1)} MW`, n]} />
                <Legend wrapperStyle={{ fontSize: 13 }} />
                <Line dataKey="locked" name="Declared" stroke="#0b7f7a" strokeWidth={2} strokeDasharray="5 4" dot={false} isAnimationActive={false} />
                <Line dataKey="proposed" name="Proposed" stroke="#9aa7b6" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                {result && <Line dataKey="applied" name="What actually applies" stroke="#13243b" strokeWidth={2} dot={false} isAnimationActive={false} />}
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </section>
  );
}
