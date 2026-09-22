"""
battery_optimizer.py — Battery dispatch LP (roadmap P1.5 rewrite).
----------------------------------------------------------------------------

WHY THIS WAS REWRITTEN:
  The previous version had no round-trip efficiency (energy in = energy
  out, physically impossible), hardcoded 1-hour blocks with no explicit
  Δt, let SOC hit 0 (real batteries reserve a floor to protect cell
  life), and its objective minimized UNMET DEMAND ONLY — so charging was
  entirely unconstrained by the objective and the LP was degenerate; CBC
  just returned whichever feasible solution it found first. It also
  optimized against a "demand" forecast, when a solar IPP has a DECLARED
  SCHEDULE, not demand (roadmap P1.6).

WHAT CHANGED:
  - Δt is explicit (default 0.25h / 15-minute blocks, matching the P1.2
    96-block grid) — every MW rate is multiplied by Δt to get MWh, not
    silently assumed to be 1 hour.
  - Separate charge/discharge efficiency (round-trip efficiency split
    symmetrically by default: each leg = sqrt(round_trip)). Energy
    ACTUALLY STORED per charge action is less than energy drawn from
    surplus; energy the battery must GIVE UP per discharge action is
    more than what's delivered to the grid.
  - SOC floor/ceiling as % of usable capacity (defaults 10%/95% — modern
    Li-ion figures, not vendor- or site-specific).
  - A terminal SOC constraint: the battery must end the horizon at (or
    above) a target level, by default its own starting level — otherwise
    an optimizer minimizing only within-horizon cost can drain the
    battery to zero right before the horizon ends and look artificially
    good.
  - A cycling cost term (small, provisional Rs/MWh-equivalent penalty
    per MWh charged+discharged) — without it, the LP has no reason to
    avoid pointless back-and-forth cycling that doesn't help.
  - The objective is DSM EXPOSURE against the declared schedule —
    |actual grid delivery - declared schedule| in every block, penalized
    in BOTH directions (under- and over-delivery), not "unmet demand"
    only. This removes the degeneracy: every unit of deviation now costs
    something.
  - Charge and discharge are now mutually exclusive via a binary
    variable (MILP, not pure LP) — the old sum-bound constraint
    (`charge + discharge <= max(rate)`) didn't actually prevent a
    physically impossible simultaneous charge+discharge, it just capped
    their sum.

DSM MODE (roadmap P1.7): `deviation_penalty_per_mwh` above is a flat
  provisional cost — fine for a plant with no configured regulatory data.
  When `dsm_contract_rate_rs_per_kwh` and `dsm_available_capacity_mw` are
  BOTH supplied, the objective switches to the REAL tiered CERC
  (Deviation Settlement Mechanism and Related Matters) Regulations, 2024
  Regulation 8(4) WS-seller structure instead (see
  src/regulatory/dsm.py): deviation is split into three volume-limit
  bands (Note-1) with DIFFERENT Rs/MWh rates for over- vs under-injection,
  not one flat number. This is exactly LP-representable with plain
  continuous segment variables and no extra binaries for the rate
  structure itself: the under-injection (payable) rates increase per band
  (100% -> 110% -> 200% of contract rate), a CONVEX cost a minimizing LP
  fills cheapest-segment-first on its own; the over-injection (receivable)
  rates decrease per band (100% -> 90% -> 0%), a CONCAVE revenue a
  cost-minimizing LP (revenue enters negated) fills highest-rate-segment-
  first on its own — both are the "good" cases for a segment
  decomposition. One binary per block (`is_over_injecting`) IS still
  needed, not for the rates but to stop the LP reporting a physically
  impossible simultaneous over- and under-injection in the same block:
  VLwS(1)'s rate is IDENTICAL for both directions (100% either way), so
  inflating both segment-1 variables together is objective-value-neutral,
  and an unconstrained LP could report a nonsensical split even though
  the true net deviation is unambiguous.
"""

import datetime as _dt

import numpy as np
import pandas as pd
from loguru import logger

try:
    import pulp
except ImportError:
    raise ImportError(
        "PuLP not installed. Run: pip install pulp"
    )

from src.regulatory import dsm as _dsm


class BatteryOptimizer:
    def __init__(
        self,
        battery_capacity_mwh: float = 50.0,
        charge_rate_mw: float = 25.0,
        discharge_rate_mw: float = 25.0,
        initial_charge_mwh: float = 25.0,
        dt_hours: float = 0.25,
        round_trip_efficiency: float = 0.90,
        soc_floor_pct: float = 0.10,
        soc_ceiling_pct: float = 0.95,
        terminal_soc_target_mwh: float | None = None,
        cycling_cost_per_mwh: float = 0.5,
        deviation_penalty_per_mwh: float = 1.0,
        dsm_contract_rate_rs_per_kwh: float | None = None,
        dsm_available_capacity_mw: float | None = None,
        dsm_seller_category: str = "solar",
        dsm_as_of: _dt.date | None = None,
    ):
        self.capacity        = battery_capacity_mwh
        self.charge_rate     = charge_rate_mw
        self.discharge_rate  = discharge_rate_mw
        self.initial_charge  = initial_charge_mwh
        self.dt_hours        = dt_hours

        # Split round-trip efficiency symmetrically across the two legs
        # (a common convention when a vendor only quotes one number):
        # charge_eff * discharge_eff = round_trip_efficiency.
        self.charge_efficiency    = np.sqrt(round_trip_efficiency)
        self.discharge_efficiency = np.sqrt(round_trip_efficiency)

        self.soc_floor_mwh    = soc_floor_pct * battery_capacity_mwh
        self.soc_ceiling_mwh  = soc_ceiling_pct * battery_capacity_mwh
        if not (self.soc_floor_mwh <= initial_charge_mwh <= self.soc_ceiling_mwh):
            raise ValueError(
                f"initial_charge_mwh={initial_charge_mwh} is outside the "
                f"usable SOC range [{self.soc_floor_mwh:.2f}, {self.soc_ceiling_mwh:.2f}] MWh "
                f"(floor {soc_floor_pct:.0%} / ceiling {soc_ceiling_pct:.0%} of capacity)"
            )

        self.terminal_soc_target = (
            terminal_soc_target_mwh if terminal_soc_target_mwh is not None
            else initial_charge_mwh
        )
        self.cycling_cost_per_mwh      = cycling_cost_per_mwh
        self.deviation_penalty_per_mwh = deviation_penalty_per_mwh

        # DSM mode (roadmap P1.7) is opt-in: both a real contract rate and
        # a real available capacity must be supplied, or this falls back
        # to the flat deviation_penalty_per_mwh above unchanged.
        self.dsm_enabled = (
            dsm_contract_rate_rs_per_kwh is not None and dsm_available_capacity_mw is not None
        )
        self.dsm_contract_rate_rs_per_kwh = dsm_contract_rate_rs_per_kwh
        self.dsm_available_capacity_mw    = dsm_available_capacity_mw
        self.dsm_seller_category          = dsm_seller_category
        self.dsm_as_of                    = dsm_as_of or _dt.date.today()
        if self.dsm_enabled:
            # Available Capacity per time block, in MWh (see
            # src/regulatory/dsm.py's UNITS note).
            self.dsm_available_capacity_mwh = dsm_available_capacity_mw * dt_hours
            self.dsm_segments = _dsm.lp_cost_segments(
                available_capacity_mwh=self.dsm_available_capacity_mwh,
                contract_rate_rs_per_kwh=dsm_contract_rate_rs_per_kwh,
                category=dsm_seller_category,
                as_of=self.dsm_as_of,
            )

        logger.info(
            f"BatteryOptimizer initialised: capacity={self.capacity}MWh, "
            f"dt={self.dt_hours}h, round_trip_eff={round_trip_efficiency:.0%}, "
            f"usable SOC=[{self.soc_floor_mwh:.1f}, {self.soc_ceiling_mwh:.1f}] MWh"
            + (f", DSM mode ON (contract_rate={dsm_contract_rate_rs_per_kwh} Rs/kWh, "
               f"category={dsm_seller_category}, as_of={self.dsm_as_of})" if self.dsm_enabled else "")
        )

    def optimize(
        self,
        solar_forecast: np.ndarray,
        declared_schedule_mw: np.ndarray,
    ) -> pd.DataFrame:
        n = len(solar_forecast)
        assert len(declared_schedule_mw) == n

        logger.info(f"Optimizing {n}-block dispatch schedule (Δt={self.dt_hours}h)")

        prob = pulp.LpProblem("battery_dispatch", pulp.LpMinimize)

        charge = [pulp.LpVariable(f"charge_{t}", 0, self.charge_rate) for t in range(n)]
        discharge = [pulp.LpVariable(f"discharge_{t}", 0, self.discharge_rate) for t in range(n)]
        battery_level = [
            pulp.LpVariable(f"battery_{t}", self.soc_floor_mwh, self.soc_ceiling_mwh)
            for t in range(n)
        ]
        # Mutual exclusivity: charging and discharging in the same block is
        # physically impossible, not just rate-limited.
        is_charging = [pulp.LpVariable(f"is_charging_{t}", cat="Binary") for t in range(n)]

        cycling_term = self.cycling_cost_per_mwh * pulp.lpSum(
            charge[t] + discharge[t] for t in range(n)
        ) * self.dt_hours

        if self.dsm_enabled:
            # Real tiered CERC Regulation 8(4) structure (see this file's
            # module docstring and src/regulatory/dsm.py) instead of the
            # flat deviation_penalty_per_mwh.
            widths = self.dsm_segments["segment_widths_mwh"]
            over_rates = self.dsm_segments["over_injection_rates_rs_per_mwh"]
            under_rates = self.dsm_segments["under_injection_rates_rs_per_mwh"]
            # A safe Big-M for the mutual-exclusivity constraints below --
            # derived from the actual data passed in, not a magic constant.
            big_m = 10 * (
                float(np.max(solar_forecast)) + float(np.max(declared_schedule_mw))
                + self.charge_rate + self.discharge_rate + 1.0
            )

            over1 = [pulp.LpVariable(f"over1_{t}", 0, widths[0]) for t in range(n)]
            over2 = [pulp.LpVariable(f"over2_{t}", 0, widths[1]) for t in range(n)]
            over3 = [pulp.LpVariable(f"over3_{t}", 0, None) for t in range(n)]
            under1 = [pulp.LpVariable(f"under1_{t}", 0, widths[0]) for t in range(n)]
            under2 = [pulp.LpVariable(f"under2_{t}", 0, widths[1]) for t in range(n)]
            under3 = [pulp.LpVariable(f"under3_{t}", 0, None) for t in range(n)]
            # Not needed for the rate structure (see docstring) -- only to
            # keep the reported split physically unambiguous.
            is_over_injecting = [pulp.LpVariable(f"is_over_{t}", cat="Binary") for t in range(n)]

            dsm_term = pulp.lpSum(
                under1[t] * under_rates[0] + under2[t] * under_rates[1] + under3[t] * under_rates[2]
                - (over1[t] * over_rates[0] + over2[t] * over_rates[1] + over3[t] * over_rates[2])
                for t in range(n)
            )
            prob += dsm_term + cycling_term
        else:
            # |grid_delivered - declared_schedule| via the standard LP split.
            deviation = [pulp.LpVariable(f"deviation_{t}", 0, None) for t in range(n)]
            prob += (
                self.deviation_penalty_per_mwh * pulp.lpSum(deviation) * self.dt_hours
                + cycling_term
            )

        for t in range(n):
            solar = float(solar_forecast[t])
            schedule = float(declared_schedule_mw[t])
            surplus = solar - schedule

            # SOC continuity, in MWh: charge STORES less than drawn (losses),
            # discharge REMOVES more than delivered (losses).
            energy_in = charge[t] * self.dt_hours * self.charge_efficiency
            energy_out = discharge[t] * self.dt_hours / self.discharge_efficiency
            prev_level = self.initial_charge if t == 0 else battery_level[t - 1]
            prob += battery_level[t] == prev_level + energy_in - energy_out

            # Charge only from solar surplus (this BESS is colocated with
            # the plant, not grid-charging) -- same assumption as before.
            prob += charge[t] <= max(0.0, surplus)

            # Mutual exclusivity via the binary.
            prob += charge[t] <= self.charge_rate * is_charging[t]
            prob += discharge[t] <= self.discharge_rate * (1 - is_charging[t])

            # DSM exposure: actual delivered vs. declared schedule.
            grid_delivered = solar + discharge[t] - charge[t]
            if self.dsm_enabled:
                # over/under segments are block ENERGY (MWh, matching
                # dsm.lp_cost_segments' available_capacity_mwh convention)
                # -- grid_delivered/schedule are MW rates, so convert.
                net_deviation_mwh = (over1[t] + over2[t] + over3[t]) - (under1[t] + under2[t] + under3[t])
                prob += net_deviation_mwh == (grid_delivered - schedule) * self.dt_hours
                prob += over1[t] + over2[t] + over3[t] <= big_m * is_over_injecting[t]
                prob += under1[t] + under2[t] + under3[t] <= big_m * (1 - is_over_injecting[t])
            else:
                prob += deviation[t] >= grid_delivered - schedule
                prob += deviation[t] >= schedule - grid_delivered

        # Terminal SOC constraint -- don't let the optimizer drain the
        # battery right at the edge of the horizon to look better within it.
        prob += battery_level[n - 1] >= self.terminal_soc_target

        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        status = pulp.LpStatus[prob.status]
        logger.info(f"Optimization status: {status}")

        if status != "Optimal":
            logger.warning(f"Non-optimal solution: {status}. Using rule-based fallback.")
            return self._rule_based_fallback(solar_forecast, declared_schedule_mw)

        results = []
        for t in range(n):
            c = max(0.0, pulp.value(charge[t]) or 0.0)
            d = max(0.0, pulp.value(discharge[t]) or 0.0)
            bl = pulp.value(battery_level[t]) or 0.0
            s = float(solar_forecast[t])
            sched = float(declared_schedule_mw[t])
            grid_delivered = s + d - c

            action = "CHARGE" if c > 0.5 else ("DISCHARGE" if d > 0.5 else "HOLD")

            row = {
                "solar_mw": round(s, 2),
                "declared_schedule_mw": round(sched, 2),
                "surplus_mw": round(s - sched, 2),
                "charge_mw": round(c, 2),
                "discharge_mw": round(d, 2),
                "battery_level_mwh": round(bl, 2),
                "grid_balance_mw": round(grid_delivered - sched, 2),
                "deviation_mwh": round(abs(grid_delivered - sched) * self.dt_hours, 3),
                "action": action,
            }
            if self.dsm_enabled:
                # Recomputed directly from src/regulatory/dsm.py (the same
                # function tests/test_dsm.py verifies against the
                # regulation text) rather than read back from the solved
                # segment variables -- one source of truth for what a
                # block's deviation actually costs.
                signed_deviation_mwh = (grid_delivered - sched) * self.dt_hours
                settlement = _dsm.deviation_settlement(
                    signed_deviation_mwh, self.dsm_available_capacity_mwh,
                    self.dsm_contract_rate_rs_per_kwh, self.dsm_seller_category, self.dsm_as_of,
                )
                row["dsm_rs"] = round(settlement["net_rs"], 2)
            results.append(row)

        return pd.DataFrame(results)

    def _rule_based_fallback(
        self,
        solar_forecast: np.ndarray,
        declared_schedule_mw: np.ndarray,
    ) -> pd.DataFrame:
        """Safety net if the LP is infeasible. Respects Δt, efficiency and
        SOC bounds, but doesn't optimize cycling cost or terminal SOC --
        it's a fallback, not the primary path."""
        logger.info("Using rule-based fallback dispatch")
        results = []
        battery = self.initial_charge

        for t in range(len(solar_forecast)):
            s = float(solar_forecast[t])
            sched = float(declared_schedule_mw[t])
            surplus = s - sched

            charge_amt = 0.0
            discharge_amt = 0.0

            if surplus > 0:
                headroom_mwh = self.soc_ceiling_mwh - battery
                max_charge_mw_for_headroom = headroom_mwh / (self.dt_hours * self.charge_efficiency) if headroom_mwh > 0 else 0.0
                charge_amt = min(surplus, self.charge_rate, max_charge_mw_for_headroom)
                battery += charge_amt * self.dt_hours * self.charge_efficiency
                action = "CHARGE" if charge_amt > 0.5 else "HOLD"
            elif surplus < 0:
                available_mwh = battery - self.soc_floor_mwh
                max_discharge_mw_for_available = (
                    available_mwh * self.discharge_efficiency / self.dt_hours if available_mwh > 0 else 0.0
                )
                discharge_amt = min(abs(surplus), self.discharge_rate, max_discharge_mw_for_available)
                battery -= discharge_amt * self.dt_hours / self.discharge_efficiency
                action = "DISCHARGE" if discharge_amt > 0.5 else "HOLD"
            else:
                action = "HOLD"

            grid_delivered = s + discharge_amt - charge_amt
            row = {
                "solar_mw": round(s, 2),
                "declared_schedule_mw": round(sched, 2),
                "surplus_mw": round(surplus, 2),
                "charge_mw": round(charge_amt, 2),
                "discharge_mw": round(discharge_amt, 2),
                "battery_level_mwh": round(battery, 2),
                "grid_balance_mw": round(grid_delivered - sched, 2),
                "deviation_mwh": round(abs(grid_delivered - sched) * self.dt_hours, 3),
                "action": action,
            }
            if self.dsm_enabled:
                signed_deviation_mwh = (grid_delivered - sched) * self.dt_hours
                settlement = _dsm.deviation_settlement(
                    signed_deviation_mwh, self.dsm_available_capacity_mwh,
                    self.dsm_contract_rate_rs_per_kwh, self.dsm_seller_category, self.dsm_as_of,
                )
                row["dsm_rs"] = round(settlement["net_rs"], 2)
            results.append(row)

        return pd.DataFrame(results)
