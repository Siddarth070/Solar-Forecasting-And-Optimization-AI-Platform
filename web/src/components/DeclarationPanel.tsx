import type { Forecast, Plant } from "../lib/api";
import { STRATEGIES, blockClock, downloadText, scheduleCsv, type LossSummary, type Outcome, type Strategy } from "../lib/blocks";
import { inr } from "../lib/format";

export type LossMatrix = Record<Strategy, Record<Outcome, LossSummary>>;

const OUTCOMES: { id: Outcome; label: string }[] = [
  { id: "p10", label: "Low day (every block at P10)" },
  { id: "p50", label: "Median day (P50)" },
  { id: "p90", label: "High day (every block at P90)" },
];

export function DeclarationPanel(props: {
  plant: Plant;
  date: string;
  forecast: Forecast;
  declared: number[];
  strategy: Strategy;
  onStrategy: (s: Strategy) => void;
  matrix: LossMatrix | null;
  matrixError: string | null;
}) {
  const { plant, date, forecast, declared, strategy, onStrategy, matrix, matrixError } = props;
  const rate = plant.regulatory.contract_rate_rs_per_kwh;

  const worst = (s: Strategy) => (matrix ? Math.max(...OUTCOMES.map((o) => matrix[s][o.id].lossRs)) : 0);
  const bestByWorstCase = matrix
    ? STRATEGIES.reduce((a, b) => (worst(b.id) < worst(a.id) ? b : a)).id
    : null;

  return (
    <section className="panel" aria-labelledby="decl-h">
      <header>
        <div>
          <h2 id="decl-h">Declaration and what it risks</h2>
          <p>
            DSM loss is what deviation settlement costs compared with having declared exactly what you delivered.
            Small deviations settle at the contract rate and cost nothing; the loss comes from blocks outside the
            first volume band.
          </p>
        </div>
        <button
          className="btn secondary"
          onClick={() => downloadText(`zenith_${plant.plant_id}_${date}_schedule.csv`, scheduleCsv(date, plant.plant_id, forecast, declared))}
        >
          Download schedule CSV
        </button>
      </header>

      {rate == null && (
        <p className="notice error">This plant has no contract rate configured, so DSM losses can't be priced.</p>
      )}
      {plant.simulated && rate != null && (
        <p className="notice" style={{ marginBottom: 16 }}>
          Priced at ₹{rate.toFixed(2)}/kWh, a placeholder contract rate for this simulated plant. Replace it with
          the real PPA tariff in configs/plants/{plant.plant_id}.yaml before relying on these figures.
        </p>
      )}

      <div className="split">
        <fieldset className="choice-list" style={{ border: 0, padding: 0, margin: 0 }}>
          <legend className="small muted" style={{ marginBottom: 8 }}>What you declare</legend>
          {STRATEGIES.map((s) => (
            <label className="choice" key={s.id}>
              <input type="radio" name="strategy" value={s.id} checked={strategy === s.id} onChange={() => onStrategy(s.id)} />
              <span className="text">
                <span>{s.label}</span>
                <span className="note">{s.note}</span>
              </span>
            </label>
          ))}
        </fieldset>

        <div>
          {matrixError && <p className="notice error">{matrixError}</p>}
          {!matrix && !matrixError && <p className="loading">Pricing nine declaration-and-outcome pairs…</p>}
          {matrix && (
            <>
              <div className="table-scroll">
                <table className="matrix">
                  <caption className="sr-only">DSM loss for each declaration under each outcome</caption>
                  <thead>
                    <tr>
                      <th scope="col" style={{ textAlign: "left" }}>If you declare</th>
                      {OUTCOMES.map((o) => (
                        <th scope="col" key={o.id}>{o.label}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {STRATEGIES.map((s) => (
                      <tr key={s.id} className={s.id === strategy ? "selected" : undefined}>
                        <th scope="row">{s.short}</th>
                        {OUTCOMES.map((o) => {
                          const c = matrix[s.id][o.id];
                          return (
                            <td key={o.id} className={c.lossRs > 0 ? "hot" : "cool"}>
                              {inr(c.lossRs)}
                              <span className="sub">
                                {c.blocksWithLoss === 0
                                  ? "all blocks inside band"
                                  : `${c.blocksWithLoss} block${c.blocksWithLoss > 1 ? "s" : ""} outside band`}
                              </span>
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="small muted" style={{ marginTop: 12 }}>
                {bestByWorstCase && worst(bestByWorstCase) > 0
                  ? `Lowest worst-case loss: ${STRATEGIES.find((s) => s.id === bestByWorstCase)!.label.replace("Declare", "declaring")}, at most ${inr(worst(bestByWorstCase))}.`
                  : "Every declaration keeps every block inside the first band in all three outcomes, so none of them costs anything under DSM."}{" "}
                {matrix[strategy].p10.worstBlock !== null &&
                  `With your current choice, a low day hurts most at ${blockClock(matrix[strategy].p10.worstBlock!)}.`}
              </p>
              <p className="small muted" style={{ marginTop: 6 }}>
                Low and high days put every block at its P10 or P90 at once. Real days mix, so treat these columns as
                bounds rather than likely outcomes.
              </p>
            </>
          )}
        </div>
      </div>
    </section>
  );
}
