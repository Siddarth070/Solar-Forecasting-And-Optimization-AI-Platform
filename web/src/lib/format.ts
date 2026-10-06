const inrFmt = new Intl.NumberFormat("en-IN", { maximumFractionDigits: 0 });

/** ₹ with Indian digit grouping (1,23,456). */
export const inr = (n: number) => `₹${inrFmt.format(Math.round(n))}`;

export const mw = (n: number, digits = 1) => `${n.toFixed(digits)} MW`;
export const mwh = (n: number, digits = 1) => `${n.toLocaleString("en-IN", { maximumFractionDigits: digits })} MWh`;

export function longDate(date: string): string {
  return new Date(`${date}T12:00:00+05:30`).toLocaleDateString("en-IN", {
    weekday: "long", day: "numeric", month: "long", year: "numeric", timeZone: "Asia/Kolkata",
  });
}
