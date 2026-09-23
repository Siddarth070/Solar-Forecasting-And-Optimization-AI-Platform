"""
time_blocks.py — The 96-block/day time grid used for Indian power
scheduling and settlement, and a resolution-independent way to get onto it.
----------------------------------------------------------------------------

WHY THIS EXISTS (roadmap P1.2):
  Indian grid scheduling and DSM settlement operate on a 96-block day: IST,
  15 minutes per block, block 1 = 00:00-00:15, block 96 = 23:45-24:00. Any
  code that ingests real metering data at a DIFFERENT resolution (1-min,
  5-min, 10-min smart meters are all common) and just calls
  `.resample("15min").mean()` gets it wrong whenever the samples aren't
  aligned to block boundaries -- a block's true energy is the INTEGRAL of
  power over that exact 15-minute window, not the mean of whichever sample
  points happen to fall near it.

WHAT THIS MODULE DOES:
  Treats power(t) as a piecewise-linear curve through the actual sample
  timestamps (whatever resolution they're at), interpolates the exact
  power at each of the 97 block-boundary instants, and trapezoidally
  integrates within each block. This is resolution-independent BY
  CONSTRUCTION: feed it the same underlying curve sampled at 1, 5, 10, or
  15-minute resolution and it converges to the same 96 block-energy
  totals (see tests/test_time_blocks.py, which checks this against a
  synthetic series with a known analytical integral).

A REAL NUMERICAL LIMIT, NOT A BUG:
  When the input is sampled at EXACTLY the block width (native 15-minute
  data), each block has only its own two boundary values to integrate
  from -- trapezoidal integration between two known points is the best
  unbiased estimate possible with zero information about what happens in
  between. For a curve with real curvature (e.g. a diurnal sine-like
  shape), the single worst block -- near where the slope changes fastest,
  not near the peak -- can exceed the roadmap's 0.1% figure even though
  every oversampled resolution (1/5/10-min feeding 15-min blocks) clears
  it comfortably, and the TOTAL daily energy and the MEAN per-block error
  both stay far under it at every resolution including native 15-min (see
  tests/test_time_blocks.py, which checks and documents this explicitly
  rather than picking an easier test to hide it).

WHAT THIS MODULE DOES NOT DO YET:
  Rewire the existing hourly-resolution pipeline (jaipur_simulator,
  build_features, train.py) onto this grid -- that migration belongs to
  the roadmap tasks that actually consume it (P1.3 probabilistic
  forecasts, P1.4 rolling horizons, P1.5 optimizer rewrite), not this one.
  P1.2's own acceptance criterion is the primitive and its test, which is
  what this file and its test provide.
"""

import numpy as np
import pandas as pd

BLOCK_MINUTES = 15
BLOCKS_PER_DAY = 96


def block_boundaries(date, tz: str = "Asia/Kolkata") -> pd.DatetimeIndex:
    """The 97 boundary timestamps (block edges) for one calendar date, IST.
    Block i (1-indexed, 1..96) spans [boundaries[i-1], boundaries[i])."""
    start = pd.Timestamp(date, tz=tz).normalize()
    return pd.date_range(start, periods=BLOCKS_PER_DAY + 1, freq=f"{BLOCK_MINUTES}min")


def block_of(timestamp) -> int:
    """1-indexed block number (1-96) for a timestamp, within its own
    calendar day (i.e. relative to that day's local midnight)."""
    ts = pd.Timestamp(timestamp)
    midnight = ts.normalize()
    minutes_since_midnight = (ts - midnight).total_seconds() / 60
    block = int(minutes_since_midnight // BLOCK_MINUTES) + 1
    return min(max(block, 1), BLOCKS_PER_DAY)


def integrate_to_blocks(series: pd.Series, date, tz: str = "Asia/Kolkata") -> pd.Series:
    """
    Time-weighted integration of a power series onto the 96-block grid.

    Parameters
    ----------
    series : pd.Series
        Power in MW, indexed by tz-aware timestamps. Resolution may be
        anything (1-min, 5-min, 10-min, 15-min, irregular) -- must cover
        at least [00:00, 24:00) of `date`, with >= 2 points in that range.
    date : anything pd.Timestamp accepts (e.g. "2024-06-01")
        The calendar date (IST) to integrate.
    tz : str
        Timezone `date` is interpreted in (must match `series`'s tz).

    Returns
    -------
    pd.Series
        Length 96, index 1..96, energy in MWh per block. Computed via
        trapezoidal integration of the power curve through the ACTUAL
        sample points plus the 97 block-boundary instants (power at each
        boundary found by linear interpolation of the real samples) --
        never a naive resample().mean() on misaligned timestamps.
    """
    edges = block_boundaries(date, tz)
    day_start, day_end = edges[0], edges[-1]

    s = series.sort_index()
    s = s[(s.index >= day_start) & (s.index <= day_end)]
    if len(s) < 2:
        raise ValueError(
            f"Need at least 2 samples within [{day_start}, {day_end}] to "
            f"integrate; got {len(s)}."
        )

    x_samples = (s.index - day_start).total_seconds().to_numpy()
    y_samples = s.to_numpy(dtype=float)
    edge_seconds = (edges - day_start).total_seconds().to_numpy()  # 97 values

    # Every point we'll integrate over: the real samples plus the 97 block
    # edges (interpolated), deduplicated and sorted -- this is what makes
    # the result exact at block boundaries regardless of input resolution.
    all_x = np.union1d(x_samples, edge_seconds)
    all_y = np.interp(all_x, x_samples, y_samples)

    block_energy_mwh = np.zeros(BLOCKS_PER_DAY)
    for i in range(BLOCKS_PER_DAY):
        lo, hi = edge_seconds[i], edge_seconds[i + 1]
        mask = (all_x >= lo) & (all_x <= hi)
        xs, ys = all_x[mask], all_y[mask]
        # Power (MW) integrated over seconds -> MWh (divide by 3600 s/h).
        block_energy_mwh[i] = np.trapezoid(ys, xs) / 3600.0

    return pd.Series(block_energy_mwh, index=pd.RangeIndex(1, BLOCKS_PER_DAY + 1, name="block"))
