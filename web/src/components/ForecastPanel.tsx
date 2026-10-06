import { Area, CartesianGrid, ComposedChart, Legend, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { Forecast, Plant } from "../lib/api";
import { blockClock, istClock } from "../lib/blocks";
import { mw, mwh } from "../lib/format";
import type { DayWeather } from "../lib/weather";

export const TICKS = [0, 12, 24, 36, 48, 60, 72, 84, 95];

export function ForecastPanel({ forecast, declared, plant, weather }: {
  forecast: Forecast;
  declared: number[];
  plant: Plant;
  weather: DayWeather;
}) {
  const data = forecast.predictions_mw.map((p, i) => ({
    i,
    point: p,
    range: forecast.predictions_p10_mw.length ? [forecast.predictions_p10_mw[i], forecast.predictions_p90_mw[i]] : undefined,
    declared: declared[i],
  }));
  const daylight = forecast.predictions_mw.filter((v) => v > 0.5).length;
  const cf = forecast.total_generation_mwh / (plant.ac_capacity_mw * 24);

  return (
    <section className="panel" aria-labelledby="fc-h">
      <header>
        <div>
          <h2 id="fc-h">Forecast</h2>
          <p>Expected output per 15-minute block, IST. The shaded band is the P10–P90 range.</p>
        </div>
      </header>

      {weather.quality.source === "clear-sky demo" ? (
        <p className="notice" style={{ marginBottom: 16 }}>
          Running on clear-sky demo weather, a cloudless day computed from sun position. This is not a forecast of
          real conditions; switch the weather source to Open-Meteo for that.
        </p>
      ) : weather.quality.notes.length ? (
        <p className="notice info" style={{ marginBottom: 16 }}>
          Weather data quality: {weather.quality.notes.join("; ")}.
        </p>
      ) : null}

      <div className="stats">
        <div className="stat"><span className="v">{mwh(forecast.total_generation_mwh)}</span><span className="k">Expected energy</span></div>
        <div className="stat"><span className="v">{mw(forecast.peak_output_mw)}</span><span className="k">Peak, at {istClock(forecast.peak_timestamp)}</span></div>
        <div className="stat"><span className="v">{(cf * 100).toFixed(1)}%</span><span className="k">Capacity factor</span></div>
        <div className="stat"><span className="v">{daylight}</span><span className="k">Generating blocks</span></div>
      </div>

      <div className="chart-box" role="img" aria-label={`Forecast chart. Peak ${forecast.peak_output_mw.toFixed(1)} MW.`}>
        <ResponsiveContainer>
          <ComposedChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -12 }}>
            <CartesianGrid stroke="#e6ebf0" vertical={false} />
            <XAxis dataKey="i" ticks={TICKS} tickFormatter={(i: number) => (i === 95 ? "24:00" : blockClock(i))} tick={{ fontSize: 12, fill: "#6a7a8f" }} />
            <YAxis domain={[0, plant.ac_capacity_mw]} tick={{ fontSize: 12, fill: "#6a7a8f" }} unit=" MW" width={72} />
            <Tooltip
              labelFormatter={(i: number) => `Block ${i + 1}, ${blockClock(i)}`}
              formatter={(v: number | number[], name: string) =>
                Array.isArray(v) ? [`${v[0].toFixed(1)}–${v[1].toFixed(1)} MW`, name] : [`${v.toFixed(1)} MW`, name]
              }
            />
            <Legend wrapperStyle={{ fontSize: 13 }} />
            <Area dataKey="range" name="P10–P90" stroke="none" fill="#e8a010" fillOpacity={0.22} isAnimationActive={false} />
            <Line dataKey="point" name="Point forecast" stroke="#e8a010" strokeWidth={2} dot={false} isAnimationActive={false} />
            <Line dataKey="declared" name="Declared" stroke="#0b7f7a" strokeWidth={2} strokeDasharray="5 4" dot={false} isAnimationActive={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}
