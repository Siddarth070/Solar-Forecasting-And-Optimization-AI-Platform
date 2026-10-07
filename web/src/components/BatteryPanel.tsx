import { useState } from "react";
import { Bar, CartesianGrid, ComposedChart, Legend, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, type BatteryParams, type Forecast, type OptimizeResult, type Plant } from "../lib/api";
import { blockClock, outcomeSeries, summarizeLoss, type LossSummary, type Outcome } from "../lib/blocks";
import { inr, mwh } from "../lib/format";
import { TICKS } from "./ForecastPanel";

const FIELDS: { key: keyof BatteryParams; label: string; step: number; min: number; max?: number }[] = [
  { key: "battery_capacity_mwh", label: "Energy capacity (MWh)", step: 5, min: 0 },
  { key: "initial_charge_mwh", label: "Charge at 00:00 (MWh)", step: 5, min: 0 },
  { key: "charge_rate_mw", label: "Max charge (MW)", step: 5, min: 0 },
  { key: "discharge_rate_mw", label: "Max discharge (MW)", step: 5, min: 0 },
  { key: "round_trip_efficiency", label: "Round-trip efficiency", step: 0.01, min: 0.5, max: 1 },
];

export function BatteryPanel(props: {
  plant: Plant;
  date: string;
  forecast: Forecast;
  declared: number[];
  lossWithout: Record<Outcome, LossSummary> | null;
}) {
  const { plant, date, forecast, declared, lossWithout } = props;
  const [params, setParams] = useState<BatteryParams>({
    battery_capacity_mwh: 50,
    initial_charge_mwh: 25,
    charge_rate_mw: 25,
    discharge_rate_mw: 25,
    round_trip_efficiency: 0.9,
  });
  const [outcome, setOutcome] = useState<Outcome>("p10");
  const [result, setResult] = useState<{ res: OptimizeResult; outcome: Outcome } | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const invalid =
    params.initial_charge_mwh > params.battery_capacity_mwh
      ? "Charge at 00:00 can't exceed the battery's energy capacity."
      : null;

  async function run() {
    setBusy(true);
    setError(null);
    try {
      const res = await api.optimize({ plantId: plant.plant_id, date, solarMw: outcomeSeries(forecast, outcome), declaredMw: declared, battery: params });
      setResult({ res, outcome });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  const rate = plant.regulatory.contract_rate_rs_per_kwh ?? 0;
  const withBattery = result ? summarizeLoss(result.res.schedule, rate) : null;
  const without = result && lossWithout ? lossWithout[result.outcome] : null;
  const data = result?.res.schedule.map((r, i) => ({
    i,
    soc: r.battery_level_mwh,
    flow: r.discharge_mw - r.charge_mw,
  }));

  return (
    <section className="panel" aria-labelledby="bat-h">
      <header>
        <div>
          <h2 id="bat-h">Battery dispatch</h2>
          <p>
            Charges and discharges block by block to cut DSM loss against your declaration. The optimiser sees the
            whole day's outcome in advance, so treat the saving as the most a battery could recover, not what it will.
          </p>
        </div>
      </header>

      <div className="split">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            if (!invalid) run();
          }}
        >
          <div className="param-grid">
            {FIELDS.map((f) => (
              <label className="field" key={f.key}>
                {f.label}
                <input
                  type="number"
                  min={f.min}
                  max={f.max}
                  step={f.step}
                  value={params[f.key]}
                  onChange={(e) => setParams({ ...params, [f.key]: Number(e.target.value) })}
                />
              </label>
            ))}
            <label className="field">
              Day it plays out against
              <select value={outcome} onChange={(e) => setOutcome(e.target.value as Outcome)}>
                <option value="p10">Low day (P10)</option>
                <option value="p50">Median day (P50)</option>
                <option value="p90">High day (P90)</option>
              </select>
            </label>
          </div>
          {invalid && <p className="notice error" style={{ marginTop: 12 }}>{invalid}</p>}
          <div className="row-actions">
            <button className="btn" type="submit" disabled={busy || !!invalid}>
              {busy ? "Running dispatch…" : "Run dispatch"}
            </button>
          </div>
        </form>

        <div>
          {error && <p className="notice error">{error}</p>}
          {!result && !error && (
            <p className="muted">Set the battery and run dispatch to see the schedule and what it recovers.</p>
          )}
          {result && withBattery && (
            <>
              <div className="stats">
                <div className="stat"><span className="v">{without ? inr(without.lossRs) : "—"}</span><span className="k">DSM loss without battery</span></div>
                <div className="stat"><span className="v">{inr(withBattery.lossRs)}</span><span className="k">With battery</span></div>
                <div className="stat">
                  <span className="v">{without ? inr(Math.max(0, without.lossRs - withBattery.lossRs)) : "—"}</span>
                  <span className="k">Recovered at most</span>
                </div>
                <div className="stat">
                  <span className="v">{mwh(result.res.summary.total_discharged_mwh)}</span>
                  <span className="k">Discharged ({result.res.summary.blocks_discharging} blocks)</span>
                </div>
              </div>
              {without && without.lossRs === 0 && (
                <p className="notice info" style={{ marginBottom: 12 }}>
                  There's no DSM loss to recover on this day: every block stays inside the first band without a
                  battery. Any cycling here earns nothing under DSM.
                </p>
              )}
              <div className="chart-box" style={{ height: 240 }} role="img" aria-label="Battery state of charge and flows">
                <ResponsiveContainer>
                  <ComposedChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -12 }}>
                    <CartesianGrid stroke="#e6ebf0" vertical={false} />
                    <XAxis dataKey="i" ticks={TICKS} tickFormatter={(i: number) => (i === 95 ? "24:00" : blockClock(i))} tick={{ fontSize: 12, fill: "#6a7a8f" }} />
                    <YAxis tick={{ fontSize: 12, fill: "#6a7a8f" }} width={60} />
                    <Tooltip
                      labelFormatter={(i: number) => `Block ${i + 1}, ${blockClock(i)}`}
                      formatter={(v: number, name: string) => [`${v.toFixed(2)} ${name.startsWith("Stored") ? "MWh" : "MW"}`, name]}
                    />
                    <Legend wrapperStyle={{ fontSize: 13 }} />
                    <Bar dataKey="flow" name="Discharge (+) / charge (−)" fill="#0b7f7a" isAnimationActive={false} />
                    <Line dataKey="soc" name="Stored energy" stroke="#13243b" strokeWidth={2} dot={false} isAnimationActive={false} />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            </>
          )}
        </div>
      </div>
    </section>
  );
}
