"""
dsm.py — CERC Deviation Settlement Mechanism (DSM) charges for a WS
(wind/solar) seller (roadmap P1.7).

SOURCE (read directly from the regulations themselves, not summarized from
memory or invented):
  - CERC (Deviation Settlement Mechanism and Related Matters) Regulations,
    2024, No. L-1/260/2021/CERC, dated 05.08.2024 (Gazette: Part III,
    Section 4, No. 642, dated 21.08.2024) -- Regulation 6 (computation of
    deviation %) and Regulation 8(4) + its Note-1 (WS seller charges and
    volume limits).
  - (First Amendment) Regulations, 2024, dated 17.12.2024, in force
    23.12.2024 -- amends only Regulation 8(8) (infirm power). Does not
    touch Regulation 8(4) or Note-1.
  - (Second Amendment) Regulations, 2025, dated 25.06.2025, in force
    01.07.2025 -- amends only Regulation 8(8) (infirm power). Does not
    touch Regulation 8(4) or Note-1.
  - (Third Amendment) Regulations, 2026, dated 26.05.2026 -- marked
    "DRAFT AMENDMENT" in the source document itself, i.e. NOT YET IN
    FORCE (proposed effective date 01.07.2026). Its substantive change to
    WS-seller charges (new clause 4A) only applies to future projects --
    bidding-route projects tendered on/after 01.01.2027, or non-bid
    projects with COD on/after 01.01.2029 -- and is irrelevant to an
    already-operating plant like the ones configured under
    configs/plants/. NOT implemented below. If a finalized Third
    Amendment is obtained later, check clause 4A again before assuming
    this module is still complete.

SCOPE:
  Only the WS-seller path (Regulation 8(4)) is implemented -- this
  platform models solar generators. The general-seller / RoR / MSW paths
  in Regulation 8(1)-(3), which are frequency-linked and apply to
  thermal/hydro/waste generators, are out of scope and not implemented.

A REAL, DOCUMENTED GAP (not fabricated -- see Ground Rule 4):
  Regulation 6(2)(b) defines the deviation-% denominator for a WS seller,
  for the period from 01.04.2026 onwards, as a blend
  [(X% of Available Capacity) + (100-X)% of Scheduled Generation)],
  "Provided 'X' shall be stipulated by the Commission through separate
  order(s) after public consultation." No such order is among the source
  documents this module was built from, and there is no published value
  for X to cite anywhere else. Rather than invent one, `deviation_pct`
  below always divides by Available Capacity alone (X=100) -- this is
  EXACTLY what Regulation 6(2)(a) specifies for the pre-01.04.2026
  period, and an honest, explicitly-labeled placeholder for the
  post-01.04.2026 period until CERC publishes X. Update this function,
  not the volume limits, once that order exists.

  The volume limits themselves (Note-1) ARE fully specified for both
  periods with no missing parameter, and are implemented exactly below,
  switched on the 01.04.2026 cutover date given in the regulation text.

UNITS:
  "Available Capacity" (Regulation 3(1)(g)) is a capacity RATING (MW) --
  "the cumulative capacity rating of wind turbines or solar inverters
  that are capable of generating power in a given time block". Regulation
  6's deviation-% formula divides an MWh numerator by this MW-denominated
  term; the regulation is only dimensionally consistent if "Available
  Capacity" is read as the ENERGY that rating could deliver over one time
  block (i.e. rating_MW * block_duration_hours), the same convention
  Regulation 3(1)(z)/(aa) use for "Scheduled generation"/"Scheduled
  drawal" ("in MW or MWh ex-bus" -- interchangeable given a block
  duration). Every function below therefore takes `available_capacity_mwh`
  (already block-energy, not a bare MW rating) so the caller states this
  conversion explicitly rather than this module guessing a block length.
"""

import datetime as _dt

CUTOVER_DATE = _dt.date(2026, 4, 1)  # Note-1(ii): "01.04.2026 onwards"

# Note-1: Volume Limits for a WS Seller, as a fraction of D_WS (deviation
# %). Value = (upper bound of VLwS(1), upper bound of VLwS(2)).
_VOLUME_LIMITS = {
    "solar": {"pre": (0.10, 0.15), "post": (0.05, 0.10)},
    "hybrid": {"pre": (0.10, 0.15), "post": (0.05, 0.10)},
    "wind": {"pre": (0.15, 0.20), "post": (0.10, 0.15)},
}

# Regulation 8(4): multiplier of contract rate for bands
# [within VLwS(1), VLwS(1)-VLwS(2), beyond VLwS(2)].
OVER_INJECTION_RATE_MULTIPLIERS = (1.00, 0.90, 0.00)   # receivable by seller
UNDER_INJECTION_RATE_MULTIPLIERS = (1.00, 1.10, 2.00)  # payable by seller


def volume_limits(category: str, as_of: _dt.date) -> tuple[float, float]:
    """Note-1's (VLwS(1), VLwS(2)) upper bounds, as fractions of D_WS,
    switched on the 01.04.2026 cutover date."""
    if category not in _VOLUME_LIMITS:
        raise ValueError(
            f"Unknown WS seller category {category!r} -- expected one of {sorted(_VOLUME_LIMITS)}"
        )
    period = "pre" if as_of < CUTOVER_DATE else "post"
    return _VOLUME_LIMITS[category][period]


def deviation_pct(actual_injection_mwh: float, scheduled_generation_mwh: float,
                   available_capacity_mwh: float) -> float:
    """Regulation 6(2) -- see the module docstring for the documented
    X-gap this uses for the post-01.04.2026 formula."""
    if available_capacity_mwh <= 0:
        return 0.0
    return 100 * (actual_injection_mwh - scheduled_generation_mwh) / available_capacity_mwh


def deviation_settlement(deviation_mwh: float, available_capacity_mwh: float,
                          contract_rate_rs_per_kwh: float, category: str,
                          as_of: _dt.date) -> dict:
    """Exact Rupee settlement for ONE time block's deviation (Regulation
    8(4)), split across Note-1's volume-limit bands.

    `deviation_mwh` follows Regulation 6's own sign convention: actual
    injection minus scheduled generation. Positive = over-injection
    (receivable); negative = under-injection (payable).

    Returns `net_rs` positive = payable BY the seller, negative =
    receivable BY the seller -- i.e. a cost/cash-outflow-positive
    convention, matching how a P&L line would read.
    """
    vl1_pct, vl2_pct = volume_limits(category, as_of)
    vl1_mwh = vl1_pct * available_capacity_mwh
    vl2_mwh = vl2_pct * available_capacity_mwh
    rate_rs_per_mwh = contract_rate_rs_per_kwh * 1000

    magnitude = abs(deviation_mwh)
    seg1 = min(magnitude, vl1_mwh)
    seg2 = min(max(magnitude - vl1_mwh, 0.0), vl2_mwh - vl1_mwh)
    seg3 = max(magnitude - vl2_mwh, 0.0)

    if deviation_mwh >= 0:
        mults, sign = OVER_INJECTION_RATE_MULTIPLIERS, -1
    else:
        mults, sign = UNDER_INJECTION_RATE_MULTIPLIERS, +1

    rs_by_segment = (seg1 * rate_rs_per_mwh * mults[0],
                      seg2 * rate_rs_per_mwh * mults[1],
                      seg3 * rate_rs_per_mwh * mults[2])
    net_rs = sign * sum(rs_by_segment)

    return {
        "deviation_mwh": deviation_mwh,
        "volume_limits_mwh": (vl1_mwh, vl2_mwh),
        "segment_mwh": (seg1, seg2, seg3),
        "segment_rate_multipliers": mults,
        "segment_rs": tuple(sign * r for r in rs_by_segment),
        "net_rs": net_rs,
    }


def lp_cost_segments(available_capacity_mwh: float, contract_rate_rs_per_kwh: float,
                      category: str, as_of: _dt.date) -> dict:
    """Segment WIDTHS (MWh, not cumulative) and Rs/MWh rates for both
    directions, for a convex piecewise-linear LP objective (see
    src/optimization/battery_optimizer.py's DSM mode). The under-injection
    (payable) rates increase per segment -- a convex cost, which plain
    continuous LP segment variables minimize correctly with no binaries.
    The over-injection (receivable) rates decrease per segment -- a
    concave revenue, which the SAME plain LP variables maximize correctly
    for the mirror-image reason (a maximizing LP fills the
    highest-marginal-rate segment first on its own, without being told
    to). See the optimizer's docstring for the full argument.
    """
    vl1_pct, vl2_pct = volume_limits(category, as_of)
    vl1_mwh = vl1_pct * available_capacity_mwh
    vl2_mwh = vl2_pct * available_capacity_mwh
    rate_rs_per_mwh = contract_rate_rs_per_kwh * 1000
    return {
        "segment_widths_mwh": (vl1_mwh, vl2_mwh - vl1_mwh),  # 3rd segment is unbounded
        "over_injection_rates_rs_per_mwh": tuple(
            rate_rs_per_mwh * m for m in OVER_INJECTION_RATE_MULTIPLIERS
        ),
        "under_injection_rates_rs_per_mwh": tuple(
            rate_rs_per_mwh * m for m in UNDER_INJECTION_RATE_MULTIPLIERS
        ),
    }
