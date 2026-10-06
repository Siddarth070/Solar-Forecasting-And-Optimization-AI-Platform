import { describe, expect, it } from "vitest";
import { blockLossRs, blockStart, clearSkyDay, istNowIso, istToday, scheduleCsv } from "./blocks";
import { interpolate, toBlocks } from "./weather";
import type { Forecast, OptimizeRow } from "./api";

function fakeOpenMeteo(date: string, next: string) {
  const time: string[] = [];
  for (const d of [date, next]) for (let i = 0; i < 96; i++)
    time.push(`${d}T${String(Math.floor(i / 4)).padStart(2, "0")}:${String((i % 4) * 15).padStart(2, "0")}`);
  // Radiation = index into the two-day series, so alignment is checkable.
  const ghi = time.map((_, i) => i);
  const htime = [date, next].flatMap((d) => Array.from({ length: 24 }, (_, h) => `${d}T${String(h).padStart(2, "0")}:00`));
  return {
    minutely_15: { time, shortwave_radiation: ghi, temperature_2m: time.map(() => 30), relative_humidity_2m: time.map(() => 40), wind_speed_10m: time.map(() => 3) },
    hourly: { time: htime, cloud_cover: htime.map((_, i) => (i % 2 ? 20 : 10)) },
  };
}

describe("toBlocks", () => {
  it("uses the value stamped t+15 for the block starting at t (preceding-mean convention)", () => {
    const { rows } = toBlocks(fakeOpenMeteo("2026-10-08", "2026-10-09"), "2026-10-08");
    expect(rows).toHaveLength(96);
    expect(rows[0].timestamp).toBe("2026-10-08T00:00:00+05:30");
    expect(rows[0].shortwave_radiation).toBe(1); // stamped 00:15
    expect(rows[95].shortwave_radiation).toBe(96); // next day's 00:00
  });

  it("interpolates hourly cloud to block midpoints", () => {
    const { rows } = toBlocks(fakeOpenMeteo("2026-10-08", "2026-10-09"), "2026-10-08");
    expect(rows[0].cloud_cover).toBeCloseTo(10 + 10 * (7.5 / 60), 0);
  });

  it("refuses a day with more than 5% missing radiation", () => {
    const body = fakeOpenMeteo("2026-10-08", "2026-10-09");
    (body.minutely_15.shortwave_radiation as (number | null)[]) = body.minutely_15.shortwave_radiation.map((v, i) => (i < 20 ? null : v));
    expect(() => toBlocks(body, "2026-10-08")).toThrow(/missing for/);
  });

  it("fills and reports small gaps", () => {
    const body = fakeOpenMeteo("2026-10-08", "2026-10-09");
    (body.minutely_15.shortwave_radiation as (number | null)[])[10] = null;
    const { rows, quality } = toBlocks(body, "2026-10-08");
    expect(rows[9].shortwave_radiation).toBe(10);
    expect(quality.missingFilled).toBe(1);
  });
});

describe("interpolate", () => {
  it("fills interior and edges", () => {
    expect(interpolate([null, 2, null, 6, null])).toEqual([2, 2, 4, 6, 6]);
  });
});

describe("blockLossRs", () => {
  const row = (balance: number, dsm: number): OptimizeRow => ({
    solar_mw: 0, declared_schedule_mw: 0, surplus_mw: 0, charge_mw: 0, discharge_mw: 0, battery_level_mwh: 0,
    grid_balance_mw: balance, deviation_mwh: Math.abs(balance) * 0.25, action: "HOLD", dsm_rs: dsm,
  });
  it("is zero inside the first band (settled at contract rate)", () => {
    // +4 MW for 15 min = 1 MWh over-injection, received at 2.5 Rs/kWh
    expect(blockLossRs(row(4, -2500), 2.5)).toBe(0);
  });
  it("is positive beyond the band", () => {
    // 1 MWh under-injection: 0.5 in band 1 (1.0x) + 0.5 in band 2 (1.1x) = 2625 payable
    expect(blockLossRs(row(-4, 2625), 2.5)).toBeCloseTo(125, 5);
  });
});

describe("time helpers", () => {
  it("formats IST block starts", () => {
    expect(blockStart("2026-10-08", 95)).toBe("2026-10-08T23:45:00+05:30");
  });
  it("computes IST date across the UTC midnight boundary", () => {
    expect(istToday(new Date("2026-10-06T20:00:00Z"))).toBe("2026-10-07");
    expect(istNowIso(new Date("2026-10-06T20:00:00Z"))).toBe("2026-10-07T01:30:00+05:30");
  });
  it("clear-sky demo is dark at night and bright at noon in Jaipur", () => {
    const d = clearSkyDay("2026-10-08", 26.91, 75.79);
    expect(d[8].shortwave_radiation).toBe(0); // 02:00
    expect(d[49].shortwave_radiation).toBeGreaterThan(700); // ~12:15
  });
});

describe("scheduleCsv", () => {
  it("writes 96 rows with block 96 ending at 24:00", () => {
    const n = Array.from({ length: 96 }, () => 1);
    const f = { predictions_mw: n, predictions_p10_mw: n, predictions_p50_mw: n, predictions_p90_mw: n } as Forecast;
    const lines = scheduleCsv("2026-10-08", "jaipur_100mw", f, n).trim().split("\n");
    expect(lines).toHaveLength(97);
    expect(lines[96]).toContain("23:45,24:00");
  });
});
