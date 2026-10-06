// The 96-block IST day, and the arithmetic the console does on top of
// API results. Kept free of React so it can be unit-tested (blocks.test.ts).

import type { Forecast, OptimizeRow, WeatherRow } from "./api";

export const BLOCKS = 96;
export const BLOCK_HOURS = 0.25;
export const IST = "Asia/Kolkata";

/** Today's calendar date in IST as YYYY-MM-DD, whatever the browser's timezone. */
export function istToday(now = new Date()): string {
  return new Intl.DateTimeFormat("en-CA", { timeZone: IST, year: "numeric", month: "2-digit", day: "2-digit" }).format(now);
}

export function addDays(date: string, days: number): string {
  const d = new Date(`${date}T12:00:00Z`);
  d.setUTCDate(d.getUTCDate() + days);
  return d.toISOString().slice(0, 10);
}

const pad = (n: number) => String(n).padStart(2, "0");

/** "HH:MM" for the start of block i (0-based). Block 0 = 00:00–00:15. */
export function blockClock(i: number): string {
  return `${pad(Math.floor(i / 4))}:${pad((i % 4) * 15)}`;
}

/** ISO timestamp (IST offset) for the start of block i on `date`. */
export function blockStart(date: string, i: number): string {
  return `${date}T${blockClock(i)}:00+05:30`;
}

/** Current wall-clock time in IST as an ISO string with +05:30. */
export function istNowIso(now = new Date()): string {
  const parts = new Intl.DateTimeFormat("en-GB", {
    timeZone: IST, year: "numeric", month: "2-digit", day: "2-digit",
    hour: "2-digit", minute: "2-digit", second: "2-digit", hourCycle: "h23",
  }).formatToParts(now);
  const get = (t: string) => parts.find((p) => p.type === t)?.value ?? "00";
  return `${get("year")}-${get("month")}-${get("day")}T${get("hour")}:${get("minute")}:${get("second")}+05:30`;
}

/** "HH:MM" in IST from any ISO timestamp. */
export function istClock(iso: string): string {
  return new Intl.DateTimeFormat("en-GB", { timeZone: IST, hour: "2-digit", minute: "2-digit", hourCycle: "h23" }).format(new Date(iso));
}

// ── Declaration strategies ─────────────────────────────────────

export type Strategy = "p50" | "point" | "p10";

export const STRATEGIES: { id: Strategy; label: string; short: string; note: string }[] = [
  { id: "p50", label: "Declare the median (P50)", short: "Median (P50)", note: "Half the time you'll deliver more, half the time less." },
  { id: "point", label: "Declare the point forecast", short: "Point forecast", note: "The single best-guess model output." },
  { id: "p10", label: "Declare conservatively (P10)", short: "Conservative (P10)", note: "You'll usually over-deliver; over-injection beyond the band is paid less or not at all." },
];

export function declaredFor(forecast: Forecast, strategy: Strategy): number[] {
  const series =
    strategy === "point" ? forecast.predictions_mw
    : strategy === "p10" ? forecast.predictions_p10_mw
    : forecast.predictions_p50_mw;
  // Quantile output is optional on the API side; fall back to the point
  // forecast rather than declaring zeros.
  return (series.length === forecast.predictions_mw.length ? series : forecast.predictions_mw).map(round2);
}

export type Outcome = "p10" | "p50" | "p90";

export function outcomeSeries(forecast: Forecast, outcome: Outcome): number[] {
  const s =
    outcome === "p10" ? forecast.predictions_p10_mw
    : outcome === "p90" ? forecast.predictions_p90_mw
    : forecast.predictions_p50_mw;
  return s.length ? s : forecast.predictions_mw;
}

// ── DSM loss ───────────────────────────────────────────────────

/**
 * Money lost to deviation settlement in one block, compared with having
 * scheduled exactly what was delivered.
 *
 * The API's dsm_rs is the raw settlement (positive = payable by the
 * seller, negative = receivable; src/regulatory/dsm.py). Inside the first
 * volume band a deviation settles at exactly the contract rate, so the
 * plant is no worse off than if it had scheduled correctly. The loss is
 * the settlement plus the contract value of the signed deviation:
 *   loss = dsm_rs + signed_deviation_mwh × contract_rate
 * which is 0 inside band 1 and positive beyond it.
 */
export function blockLossRs(row: OptimizeRow, contractRsPerKwh: number, dtHours = BLOCK_HOURS): number {
  if (row.dsm_rs === undefined) return 0;
  const signedMwh = row.grid_balance_mw * dtHours;
  const loss = row.dsm_rs + signedMwh * contractRsPerKwh * 1000;
  // grid_balance_mw is rounded to 0.01 MW by the API; ignore sub-rupee noise.
  return loss < 5 ? 0 : loss;
}

export interface LossSummary {
  lossRs: number;
  settlementRs: number;
  blocksWithLoss: number;
  worstBlock: number | null;
  deviationMwh: number;
}

export function summarizeLoss(rows: OptimizeRow[], contractRsPerKwh: number): LossSummary {
  let lossRs = 0, settlementRs = 0, blocksWithLoss = 0, deviationMwh = 0;
  let worstBlock: number | null = null, worst = 0;
  rows.forEach((r, i) => {
    const l = blockLossRs(r, contractRsPerKwh);
    lossRs += l;
    settlementRs += r.dsm_rs ?? 0;
    deviationMwh += r.deviation_mwh;
    if (l > 0) blocksWithLoss++;
    if (l > worst) { worst = l; worstBlock = i; }
  });
  return { lossRs, settlementRs, blocksWithLoss, worstBlock, deviationMwh };
}

// ── Export ─────────────────────────────────────────────────────

export function scheduleCsv(date: string, plantId: string, forecast: Forecast, declared: number[]): string {
  const head = "plant_id,date,block,start_ist,end_ist,forecast_mw,p10_mw,p50_mw,p90_mw,declared_mw";
  const lines = declared.map((d, i) => {
    const end = i === BLOCKS - 1 ? "24:00" : blockClock(i + 1);
    return [
      plantId, date, i + 1, blockClock(i), end,
      forecast.predictions_mw[i] ?? "", forecast.predictions_p10_mw[i] ?? "",
      forecast.predictions_p50_mw[i] ?? "", forecast.predictions_p90_mw[i] ?? "", d,
    ].join(",");
  });
  return [head, ...lines].join("\n") + "\n";
}

export function downloadText(filename: string, text: string, type = "text/csv") {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

// ── Synthetic clear-sky weather (labelled demo input) ──────────

/** Cosine of the solar zenith angle at an instant, NOAA approximation. */
export function cosZenith(utcMs: number, lat: number, lon: number): number {
  const d = new Date(utcMs);
  const start = Date.UTC(d.getUTCFullYear(), 0, 1);
  const doy = Math.floor((utcMs - start) / 86400000) + 1;
  const hourUtc = d.getUTCHours() + d.getUTCMinutes() / 60 + d.getUTCSeconds() / 3600;
  const g = ((2 * Math.PI) / 365) * (doy - 1 + (hourUtc - 12) / 24);
  const eqTime = 229.18 * (0.000075 + 0.001868 * Math.cos(g) - 0.032077 * Math.sin(g) - 0.014615 * Math.cos(2 * g) - 0.040849 * Math.sin(2 * g));
  const decl = 0.006918 - 0.399912 * Math.cos(g) + 0.070257 * Math.sin(g) - 0.006758 * Math.cos(2 * g) + 0.000907 * Math.sin(2 * g) - 0.002697 * Math.cos(3 * g) + 0.00148 * Math.sin(3 * g);
  const trueSolarMin = hourUtc * 60 + eqTime + 4 * lon;
  const ha = ((trueSolarMin / 4 - 180) * Math.PI) / 180;
  const latR = (lat * Math.PI) / 180;
  return Math.sin(latR) * Math.sin(decl) + Math.cos(latR) * Math.cos(decl) * Math.cos(ha);
}

/**
 * A cloudless day from geometry alone (Haurwitz clear-sky GHI). Not a
 * forecast — used only when the operator explicitly picks demo weather,
 * and every screen that shows it says so.
 */
export function clearSkyDay(date: string, lat: number, lon: number): WeatherRow[] {
  return Array.from({ length: BLOCKS }, (_, i) => {
    const ts = blockStart(date, i);
    const mid = new Date(ts).getTime() + 7.5 * 60000;
    const cz = cosZenith(mid, lat, lon);
    const ghi = cz > 0.01 ? 1098 * cz * Math.exp(-0.059 / cz) : 0;
    const f = Math.max(0, cz);
    return {
      timestamp: ts,
      shortwave_radiation: round1(ghi),
      cloud_cover: 5,
      temperature_2m: round1(24 + 10 * f),
      relative_humidity_2m: 40,
      wind_speed_10m: 3,
    };
  });
}

export const round1 = (n: number) => Math.round(n * 10) / 10;
export const round2 = (n: number) => Math.round(n * 100) / 100;
