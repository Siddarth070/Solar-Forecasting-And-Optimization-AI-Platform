import numpy as np
import pandas as pd
from loguru import logger

try:
    import pulp
except ImportError:
    raise ImportError(
        "PuLP not installed. Run: pip install pulp"
    )


class BatteryOptimizer:
    def __init__(
        self,
        battery_capacity_mwh: float = 50.0,
        charge_rate_mw: float = 25.0,
        discharge_rate_mw: float = 25.0,
        initial_charge_mwh: float = 25.0,
    ):
        self.capacity        = battery_capacity_mwh
        self.charge_rate     = charge_rate_mw
        self.discharge_rate  = discharge_rate_mw
        self.initial_charge  = initial_charge_mwh

        logger.info(
            f"BatteryOptimizer initialised: "
            f"capacity={self.capacity}MWh, "
            f"charge_rate={self.charge_rate}MW"
        )

    def optimize(
        self,
        solar_forecast: np.ndarray,
        demand_forecast: np.ndarray,
    ) -> pd.DataFrame:
        
        n_hours = len(solar_forecast)
        assert len(demand_forecast) == n_hours

        logger.info(f"Optimizing {n_hours}-hour dispatch schedule")

        # ── Create LP problem ─────────────────────────────────
        prob = pulp.LpProblem(
            "battery_dispatch",
            pulp.LpMinimize  # minimize grid imbalance
        )

        # ── Decision variables ────────────────────────────────
        # How much to charge each hour (0 to charge_rate)
        charge = [
            pulp.LpVariable(f"charge_{t}", 0, self.charge_rate)
            for t in range(n_hours)
        ]

        # How much to discharge each hour (0 to discharge_rate)
        discharge = [
            pulp.LpVariable(f"discharge_{t}", 0, self.discharge_rate)
            for t in range(n_hours)
        ]

        # Battery level at end of each hour
        battery_level = [
            pulp.LpVariable(f"battery_{t}", 0, self.capacity)
            for t in range(n_hours)
        ]

        # Unmet demand (shortage we couldn't cover)
        unmet = [
            pulp.LpVariable(f"unmet_{t}", 0, None)
            for t in range(n_hours)
        ]

        # ── Objective: minimize unmet demand ──────────────────
        # We want to cover as much shortage as possible
        prob += pulp.lpSum(unmet)

        # ── Constraints ───────────────────────────────────────
        for t in range(n_hours):
            surplus = float(solar_forecast[t]) - float(demand_forecast[t])

            # Battery level continuity
            if t == 0:
                prob += (
                    battery_level[t] ==
                    self.initial_charge + charge[t] - discharge[t]
                )
            else:
                prob += (
                    battery_level[t] ==
                    battery_level[t-1] + charge[t] - discharge[t]
                )

            # Can't charge more than available surplus
            prob += charge[t] <= max(0, surplus)

            # Grid balance — unmet demand
            prob += (
                unmet[t] >=
                float(demand_forecast[t]) -
                float(solar_forecast[t]) -
                discharge[t]
            )

            # Can't charge AND discharge simultaneously
            prob += charge[t] + discharge[t] <= max(
                self.charge_rate, self.discharge_rate
            )

        # ── Solve ─────────────────────────────────────────────
        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        status = pulp.LpStatus[prob.status]
        logger.info(f"Optimization status: {status}")

        if status != 'Optimal':
            logger.warning(
                f"Non-optimal solution: {status}. "
                "Using rule-based fallback."
            )
            return self._rule_based_fallback(
                solar_forecast, demand_forecast
            )

        # ── Extract results ───────────────────────────────────
        results = []
        for t in range(n_hours):
            c  = max(0, pulp.value(charge[t]) or 0)
            d  = max(0, pulp.value(discharge[t]) or 0)
            bl = max(0, pulp.value(battery_level[t]) or 0)
            s  = float(solar_forecast[t])
            dem = float(demand_forecast[t])

            # Determine action label
            if c > 0.5:
                action = "CHARGE"
            elif d > 0.5:
                action = "DISCHARGE"
            else:
                action = "HOLD"

            results.append({
                'solar_mw'       : round(s, 2),
                'demand_mw'      : round(dem, 2),
                'surplus_mw'     : round(s - dem, 2),
                'charge_mw'      : round(c, 2),
                'discharge_mw'   : round(d, 2),
                'battery_level_mwh': round(bl, 2),
                'grid_balance_mw': round(
                    s + d - c - dem, 2
                ),
                'action'         : action,
            })

        return pd.DataFrame(results)

    def _rule_based_fallback(
        self,
        solar_forecast: np.ndarray,
        demand_forecast: np.ndarray,
    ) -> pd.DataFrame:
        logger.info("Using rule-based fallback dispatch")
        results = []
        battery = self.initial_charge

        for t in range(len(solar_forecast)):
            s      = float(solar_forecast[t])
            dem    = float(demand_forecast[t])
            surplus = s - dem

            charge_amt    = 0.0
            discharge_amt = 0.0

            if surplus > 0:
                # Charge battery with surplus
                charge_amt = min(
                    surplus,
                    self.charge_rate,
                    self.capacity - battery
                )
                battery += charge_amt
                action   = "CHARGE" if charge_amt > 0.5 else "HOLD"

            elif surplus < 0:
                # Discharge battery to cover shortage
                discharge_amt = min(
                    abs(surplus),
                    self.discharge_rate,
                    battery
                )
                battery -= discharge_amt
                action   = "DISCHARGE" if discharge_amt > 0.5 else "HOLD"
            else:
                action = "HOLD"

            results.append({
                'solar_mw'         : round(s, 2),
                'demand_mw'        : round(dem, 2),
                'surplus_mw'       : round(surplus, 2),
                'charge_mw'        : round(charge_amt, 2),
                'discharge_mw'     : round(discharge_amt, 2),
                'battery_level_mwh': round(battery, 2),
                'grid_balance_mw'  : round(
                    s + discharge_amt - charge_amt - dem, 2
                ),
                'action'           : action,
            })

        return pd.DataFrame(results)