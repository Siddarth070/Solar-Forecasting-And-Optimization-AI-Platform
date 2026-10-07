import { useEffect, useState } from "react";
import { api, type ModelCard } from "../lib/api";

const NAMES: Record<string, string> = {
  model: "Zenith model",
  smart_persistence: "Smart persistence",
  persistence: "Persistence",
  physics: "Physics only",
};

const EXPLAIN: Record<string, string> = {
  smart_persistence: "Yesterday's output, scaled by today's clear-sky shape.",
  persistence: "Repeat yesterday's output.",
  physics: "Irradiance to power with fixed panel parameters, no learning.",
};

function Bars({ values }: { values: Record<string, number> }) {
  const order = ["model", "smart_persistence", "persistence", "physics"].filter((k) => k in values);
  const max = Math.max(...order.map((k) => values[k]));
  return (
    <div className="bars" role="list">
      {order.map((k) => (
        <div className={`bar-row${k === "model" ? " model" : ""}`} key={k} role="listitem">
          <span>{NAMES[k] ?? k}</span>
          <div className="track" aria-hidden>
            <div className="fill" style={{ width: `${(values[k] / max) * 100}%` }} />
          </div>
          <span className="num">{values[k].toFixed(2)}%</span>
        </div>
      ))}
    </div>
  );
}

export function Model() {
  const [card, setCard] = useState<ModelCard | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.modelCard().then(setCard).catch((e) => setError(e.message));
  }, []);

  return (
    <main>
      <section className="section">
        <div className="wrap">
          <h1 style={{ fontSize: "var(--t-xl)" }}>How the model is scored</h1>
          <p style={{ marginTop: 16 }}>
            Error is reported as nMAE, the mean absolute error as a percentage of plant capacity, because percentage
            error breaks down at dawn and dusk when output is near zero. Every figure is compared with simple
            baselines that need no machine learning, on days the model never saw in training.
          </p>
          <p className="notice" style={{ marginTop: 20 }}>
            All of these numbers come from simulated weather and a simulated plant. They show the evaluation method
            works and the model beats naive baselines on that data. They say nothing yet about accuracy on a real
            plant.
          </p>
        </div>
      </section>

      {error && (
        <div className="wrap section">
          <p className="notice error">{error}</p>
        </div>
      )}
      {!card && !error && <p className="wrap loading">Loading the model card…</p>}

      {card && (
        <>
          {card.rolling_origin_backtest && (
            <section className="section">
              <div className="wrap two-col">
                <div>
                  <h2>Walk-forward backtest</h2>
                  <p>
                    {card.rolling_origin_backtest.n_origins} forecast start points spread over a year, each predicting
                    the next {card.rolling_origin_backtest.horizon_hours} hours using only data available before it.
                    No random shuffling, so no future data leaks into training.
                  </p>
                  <p className="small muted" style={{ marginTop: 12 }}>Average nMAE, lower is better.</p>
                </div>
                <Bars values={card.rolling_origin_backtest.averaged_nmae_pct} />
              </div>
            </section>
          )}

          {card.final_holdout && (
            <section className="section">
              <div className="wrap two-col">
                <div>
                  <h2>Untouched holdout month</h2>
                  <p>
                    {card.final_holdout.period.days} days from {card.final_holdout.period.start}, scored once. The model
                    is fed day-ahead weather with realistic forecast error, the way it would run in operation.
                  </p>
                  <ul className="small muted" style={{ marginTop: 12, paddingLeft: 18 }}>
                    {Object.entries(EXPLAIN).map(([k, v]) => (
                      <li key={k}><strong>{NAMES[k]}:</strong> {v}</li>
                    ))}
                  </ul>
                </div>
                <Bars values={card.final_holdout.deliverable_skill_nmae_pct} />
              </div>
            </section>
          )}

          <section className="section">
            <div className="wrap two-col">
              <div>
                <h2>What it learns from</h2>
                <p>
                  Weather and sun position only. It doesn't use the plant's own recent output, because no real plant
                  history is connected yet. That is the first thing real meter data would add.
                </p>
              </div>
              <dl className="kv">
                <dt>Inputs</dt>
                <dd>{card.feature_list.join(", ")}</dd>
                <dt>Predicts</dt>
                <dd>Capacity factor, scaled to each plant's AC capacity</dd>
                <dt>Training data</dt>
                <dd>{card.data_window.days} simulated days from {card.data_window.start}, {card.plant_id}</dd>
                <dt>Range</dt>
                <dd>{card.quantile_model ? "P10, P50 and P90 from a separate quantile model" : "Point forecast only"}</dd>
                {card.git_sha && (
                  <>
                    <dt>Built from</dt>
                    <dd>
                      <a href={`https://github.com/Siddarth070/Solar-Forecasting-And-Optimization-AI-Platform/commit/${card.git_sha}`}>
                        commit {card.git_sha.slice(0, 7)}
                      </a>
                    </dd>
                  </>
                )}
              </dl>
            </div>
          </section>
        </>
      )}
    </main>
  );
}
