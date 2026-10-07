import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { DayStrip } from "../components/DayStrip";
import { addDays, declaredFor, istToday } from "../lib/blocks";
import { longDate, mwh } from "../lib/format";
import { useDayForecast, type WeatherSource } from "../lib/useDayForecast";

const DEMO_PLANT = "jaipur_100mw";
const PILOT_MAIL =
  "mailto:siddharthaagrawal07@gmail.com?subject=Zenith%20pilot&body=Plant%20capacity%20(MW)%3A%0AState%3A%0AWho%20prepares%20your%20day-ahead%20schedule%20today%3A%0A";

function HeroStrip() {
  const date = addDays(istToday(), 1);
  const [source, setSource] = useState<WeatherSource>("open-meteo");
  const s = useDayForecast(DEMO_PLANT, date, source);

  // If live weather is unreachable, fall back once to labelled demo weather.
  useEffect(() => {
    if (s.status === "error" && source === "open-meteo" && s.plant) setSource("demo");
  }, [s.status, s.plant, source]);

  const f = s.forecast;
  const cap = s.plant?.ac_capacity_mw ?? 100;
  const empty = Array(96).fill(0);

  let caption: React.ReactNode;
  if (f && s.plant) {
    caption = (
      <>
        <span>
          <strong>{longDate(date)}</strong>, {s.plant.name}, {s.plant.ac_capacity_mw} MW.{" "}
          {source === "demo" ? "Clear-sky demo weather (live weather unavailable)." : "Open-Meteo weather, Zenith model."}{" "}
          Expected {mwh(f.total_generation_mwh, 0)}.
        </span>
        <span>
          <span className="legend-swatch" style={{ background: "var(--sun)" }} />
          Point forecast
          <span className="legend-swatch" style={{ background: "rgba(232,160,16,.35)", marginLeft: 14 }} />
          Up to P90
          <span className="legend-swatch" style={{ background: "#5fd3cb", height: 2, marginLeft: 14 }} />
          Declared at P50
        </span>
      </>
    );
  } else if (s.status === "error") {
    caption = <span>Tomorrow's forecast appears here when the Zenith API is running. {s.error}</span>;
  } else {
    caption = <span>{s.step || "Loading tomorrow's forecast"}…</span>;
  }

  return (
    <figure className="hero-strip">
      <figcaption>{caption}</figcaption>
      <DayStrip
        values={f?.predictions_mw ?? empty}
        high={f?.predictions_p90_mw.length ? f.predictions_p90_mw : undefined}
        low={f?.predictions_p10_mw.length ? f.predictions_p10_mw : undefined}
        declared={f ? declaredFor(f, "p50") : undefined}
        max={cap}
        label={
          f
            ? `Forecast for ${date} across 96 fifteen-minute blocks, peaking at ${f.peak_output_mw.toFixed(1)} MW.`
            : "Forecast not loaded"
        }
        height={200}
      />
    </figure>
  );
}

const LEDGER: { item: string; state: "built" | "gap"; detail: string }[] = [
  { item: "96-block forecast with P10/P50/P90 range", state: "built", detail: "Gradient-boosted model on weather and solar geometry. Trained and scored on simulated Jaipur data only." },
  { item: "CERC DSM settlement for solar sellers", state: "built", detail: "DSM Regulations 2024, Reg. 8(4) bands, including the 1 April 2026 tightening. Tested against the regulation text." },
  { item: "Schedule revision timing", state: "built", detail: "IEGC 2023 Reg. 49: when a revision takes effect, and that collective sellers can't revise." },
  { item: "Battery dispatch against DSM cost", state: "built", detail: "Linear program on 15-minute blocks with round-trip losses. It assumes the battery knows the outcome, so its savings are an upper bound." },
  { item: "Real plant telemetry", state: "gap", detail: "No live plant yet. Every accuracy figure on this site comes from simulated data." },
  { item: "Forecast vs. actual, from your meter data", state: "gap", detail: "Needs a plant's 15-minute meter export. This is what a pilot builds first." },
  { item: "Your PPA tariff", state: "gap", detail: "Rupee figures use a placeholder ₹2.50/kWh contract rate until a real tariff is entered." },
];

export function Landing() {
  return (
    <main>
      <section className="hero">
        <div className="wrap">
          <h1>See what tomorrow's schedule will cost before you submit it.</h1>
          <p className="lede">
            Zenith forecasts a solar plant's output for every 15-minute block, helps you choose what to declare,
            and prices the deviation risk under CERC's settlement rules.
          </p>
          <div className="actions">
            <Link to="/console" className="btn on-dark">Open the console</Link>
            <a href={PILOT_MAIL} className="btn secondary">Ask about a pilot</a>
          </div>
          <HeroStrip />
        </div>
      </section>

      <section className="section">
        <div className="wrap">
          <h2>From forecast to a schedule you can defend</h2>
          <p className="muted">The console follows the order a scheduling desk actually works in.</p>
          <ol className="steps">
            <li>
              <h3>Forecast every block</h3>
              <p>Expected output for all 96 blocks of the day, with the range you should plan for, not just one line.</p>
            </li>
            <li>
              <h3>Choose a declaration</h3>
              <p>Compare declaring the median, the point forecast or a conservative figure, by what each loses to DSM if the day goes badly.</p>
            </li>
            <li>
              <h3>Recover what you can</h3>
              <p>See what a battery or an intraday revision could claw back, and when a revision would actually take effect.</p>
            </li>
          </ol>
        </div>
      </section>

      <section className="section">
        <div className="wrap two-col">
          <div>
            <h2>Where it stands</h2>
            <p>
              The scheduling and settlement logic is built and tested. The forecast has not yet met a real plant.
              That is the gap a pilot closes.
            </p>
            <p style={{ marginTop: 16 }}>
              <Link to="/model">How the model is scored</Link>
            </p>
          </div>
          <table className="ledger">
            <tbody>
              {LEDGER.map((r) => (
                <tr key={r.item}>
                  <th scope="row">
                    {r.item}
                    <div className={r.state === "built" ? "state-built small" : "state-gap small"}>
                      {r.state === "built" ? "Built" : "Not yet"}
                    </div>
                  </th>
                  <td>{r.detail}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="section">
        <div className="wrap two-col">
          <div>
            <h2>Looking for one pilot plant</h2>
            <p>
              A 5–100 MW plant, anywhere in India, that prepares its own day-ahead schedule or works with a QCA.
            </p>
          </div>
          <div>
            <p>
              Share three months of 15-minute meter data and the schedules you declared. Zenith will replay those
              days and show, block by block, where DSM money was lost and how much a different declaration would
              have kept. No integration and no cost for the pilot.
            </p>
            <div className="row-actions">
              <a href={PILOT_MAIL} className="btn">Ask about a pilot</a>
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}
