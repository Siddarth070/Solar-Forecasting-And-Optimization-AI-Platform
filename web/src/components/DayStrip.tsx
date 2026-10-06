import { useState } from "react";
import { BLOCKS, blockClock } from "../lib/blocks";

interface Props {
  /** One value per block, MW. */
  values: number[];
  low?: number[];
  high?: number[];
  declared?: number[];
  /** Full-scale value (plant AC capacity). */
  max: number;
  label: string;
  height?: number;
}

/**
 * The 96-block day: one column per 15-minute settlement block, IST.
 * Bar = expected output; light extension = P10–P90 range; line = declared.
 */
export function DayStrip({ values, low, high, declared, max, label, height = 180 }: Props) {
  const [hover, setHover] = useState<number | null>(null);
  const pct = (v: number | undefined) => `${Math.max(0, Math.min(100, ((v ?? 0) / max) * 100))}%`;

  return (
    <div className="day-strip" style={{ ["--strip-h" as string]: `${height}px` }}>
      <div className="cells" role="img" aria-label={label} onMouseLeave={() => setHover(null)}>
        {Array.from({ length: BLOCKS }, (_, i) => (
          <div
            key={i}
            className="cell"
            style={{ ["--i" as string]: i }}
            onMouseEnter={() => setHover(i)}
          >
            {high && <div className="range" style={{ height: pct(high[i]) }} />}
            <div className="bar" style={{ height: pct(values[i]) }} />
            {declared && <div className="declared" style={{ bottom: pct(declared[i]) }} />}
          </div>
        ))}
      </div>
      {hover !== null && (
        <div className="tip" style={{ left: `${((hover + 0.5) / BLOCKS) * 100}%` }} aria-hidden>
          <strong>Block {hover + 1}</strong>, {blockClock(hover)}–{hover === 95 ? "24:00" : blockClock(hover + 1)}
          <br />
          {values[hover]?.toFixed(1)} MW
          {low && high ? `, range ${low[hover]?.toFixed(1)}–${high[hover]?.toFixed(1)}` : ""}
          {declared ? `, declared ${declared[hover]?.toFixed(1)}` : ""}
        </div>
      )}
      <div className="axis" aria-hidden>
        {["00:00", "03:00", "06:00", "09:00", "12:00", "15:00", "18:00", "21:00"].map((t) => (
          <span key={t}>{t}</span>
        ))}
      </div>
    </div>
  );
}
