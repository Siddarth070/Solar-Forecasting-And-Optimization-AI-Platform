// Weather for one IST calendar day on the 96-block grid, from Open-Meteo.
//
// Three alignment details that are easy to get silently wrong:
//  1. Units. Open-Meteo returns wind in km/h by default; the model was
//     trained on m/s (src/ingestion/jaipur_simulator.py). We ask for m/s.
//  2. Radiation timing. Open-Meteo's 15-minute shortwave_radiation at time
//     T is the MEAN over the PRECEDING 15 minutes. The block that starts at
//     t covers [t, t+15), so its irradiance is the value stamped t+15 —
//     which means block 96 needs the next day's 00:00 value.
//  3. Timezone. We request timezone=Asia/Kolkata so returned local times
//     line up with IST block starts without any offset arithmetic.
//
// Gaps are filled by linear interpolation and reported; more than 5%
// missing (configs/config.yaml validation.max_missing_pct) is refused.

import type { WeatherRow } from "./api";
import { BLOCKS, addDays, blockClock, blockStart, round1 } from "./blocks";

export interface WeatherQuality {
  source: "open-meteo" | "clear-sky demo";
  missingFilled: number;
  clipped: number;
  notes: string[];
}

export interface DayWeather {
  rows: WeatherRow[];
  quality: WeatherQuality;
}

const MAX_MISSING_FRACTION = 0.05;

type Series = (number | null)[];

interface OpenMeteoResponse {
  minutely_15?: Record<string, Series | string[]> & { time: string[] };
  hourly?: Record<string, Series | string[]> & { time: string[] };
  reason?: string;
}

export function openMeteoUrl(date: string, lat: number, lon: number): string {
  const q = new URLSearchParams({
    latitude: String(lat),
    longitude: String(lon),
    minutely_15: "shortwave_radiation,temperature_2m,relative_humidity_2m,wind_speed_10m",
    hourly: "cloud_cover",
    timezone: "Asia/Kolkata",
    wind_speed_unit: "ms",
    start_date: date,
    end_date: addDays(date, 1),
  });
  return `https://api.open-meteo.com/v1/forecast?${q}`;
}

export async function fetchDayWeather(date: string, lat: number, lon: number): Promise<DayWeather> {
  let res: Response;
  try {
    res = await fetch(openMeteoUrl(date, lat, lon));
  } catch {
    throw new Error("Couldn't reach Open-Meteo. Check your connection, or run the console on clear-sky demo weather.");
  }
  const body = (await res.json().catch(() => ({}))) as OpenMeteoResponse;
  if (!res.ok) {
    throw new Error(`Open-Meteo refused the request: ${body.reason ?? res.statusText}. It only forecasts about 16 days ahead.`);
  }
  return toBlocks(body, date);
}

/** Pure transform from an Open-Meteo response to 96 blocks. Exported for tests. */
export function toBlocks(body: OpenMeteoResponse, date: string): DayWeather {
  const m = body.minutely_15;
  const h = body.hourly;
  if (!m?.time || !h?.time) throw new Error("Open-Meteo returned no 15-minute data for this location and date.");

  const idx15 = new Map(m.time.map((t, i) => [t, i]));
  const at15 = (key: string, local: string): number | null => {
    const i = idx15.get(local);
    const v = i === undefined ? null : (m[key] as Series)[i];
    return v ?? null;
  };
  const localStamp = (d: string, blockIndex: number) =>
    blockIndex >= BLOCKS ? `${addDays(d, 1)}T00:00` : `${d}T${blockClock(blockIndex)}`;

  // Hourly cloud cover, interpolated to each block's midpoint.
  const cloudByHour = new Map(h.time.map((t, i) => [t, (h.cloud_cover as Series)[i] ?? null]));
  const cloudAt = (i: number): number | null => {
    const hour = Math.floor(i / 4);
    const frac = ((i % 4) * 15 + 7.5) / 60;
    const a = cloudByHour.get(`${date}T${String(hour).padStart(2, "0")}:00`);
    const b = cloudByHour.get(hour + 1 >= 24 ? `${addDays(date, 1)}T00:00` : `${date}T${String(hour + 1).padStart(2, "0")}:00`);
    if (a == null) return b ?? null;
    if (b == null) return a;
    return a + (b - a) * frac;
  };

  const mid = (key: string, i: number): number | null => {
    const a = at15(key, localStamp(date, i));
    const b = at15(key, localStamp(date, i + 1));
    if (a == null) return b;
    if (b == null) return a;
    return (a + b) / 2;
  };

  const raw = {
    shortwave_radiation: Array.from({ length: BLOCKS }, (_, i) => at15("shortwave_radiation", localStamp(date, i + 1))),
    temperature_2m: Array.from({ length: BLOCKS }, (_, i) => mid("temperature_2m", i)),
    relative_humidity_2m: Array.from({ length: BLOCKS }, (_, i) => mid("relative_humidity_2m", i)),
    wind_speed_10m: Array.from({ length: BLOCKS }, (_, i) => mid("wind_speed_10m", i)),
    cloud_cover: Array.from({ length: BLOCKS }, (_, i) => cloudAt(i)),
  };

  const notes: string[] = [];
  let missingFilled = 0;
  const filled: Record<keyof typeof raw, number[]> = {} as never;
  for (const key of Object.keys(raw) as (keyof typeof raw)[]) {
    const series = raw[key];
    const missing = series.filter((v) => v == null).length;
    if (missing / BLOCKS > MAX_MISSING_FRACTION) {
      throw new Error(`${key} is missing for ${missing} of 96 blocks (limit is 5%). The forecast would be unreliable, so it wasn't run.`);
    }
    if (missing) notes.push(`${key}: ${missing} block${missing > 1 ? "s" : ""} interpolated`);
    missingFilled += missing;
    filled[key] = interpolate(series);
  }

  // Clip to the API's validated ranges (src/api/main.py WeatherInput) and say so.
  const bounds: Record<keyof typeof raw, [number, number]> = {
    shortwave_radiation: [0, 1200],
    cloud_cover: [0, 100],
    temperature_2m: [-10, 60],
    relative_humidity_2m: [0, 100],
    wind_speed_10m: [0, 50],
  };
  let clipped = 0;
  const rows: WeatherRow[] = Array.from({ length: BLOCKS }, (_, i) => {
    const row = { timestamp: blockStart(date, i) } as WeatherRow;
    for (const key of Object.keys(filled) as (keyof typeof raw)[]) {
      const [lo, hi] = bounds[key];
      const v = filled[key][i];
      const c = Math.min(hi, Math.max(lo, v));
      if (c !== v) clipped++;
      row[key] = round1(c);
    }
    return row;
  });
  if (clipped) notes.push(`${clipped} value${clipped > 1 ? "s" : ""} outside the model's valid range were clipped`);

  return { rows, quality: { source: "open-meteo", missingFilled, clipped, notes } };
}

/** Linear interpolation across nulls; edges take the nearest known value. */
export function interpolate(series: (number | null)[]): number[] {
  const out = series.slice();
  const known = out.map((v, i) => (v == null ? -1 : i)).filter((i) => i >= 0);
  if (!known.length) return out.map(() => 0);
  for (let i = 0; i < out.length; i++) {
    if (out[i] != null) continue;
    const prev = [...known].reverse().find((k) => k < i);
    const next = known.find((k) => k > i);
    if (prev === undefined) out[i] = out[next!];
    else if (next === undefined) out[i] = out[prev];
    else out[i] = (out[prev] as number) + (((out[next] as number) - (out[prev] as number)) * (i - prev)) / (next - prev);
  }
  return out as number[];
}
