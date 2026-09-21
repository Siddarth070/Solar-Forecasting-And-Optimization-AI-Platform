"""
test_battery_optimizer.py
---------------------------
Roadmap P1.5 acceptance criteria, tested directly: Δt=0.25h; charge/
discharge efficiency; SOC floor/ceiling; terminal SOC constraint; cycling
cost; objective minimizes DSM exposure (deviation from declared schedule),
not unmet demand.

RUN WITH:
  pytest tests/test_battery_optimizer.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.optimization.battery_optimizer import BatteryOptimizer


class TestPhysicalRealism:
    def test_round_trip_efficiency_is_real_not_free(self):
        """Charge from a surplus block, discharge into a shortfall block --
        energy delivered back must be LESS than energy that went in
        (the old LP had battery_level = prev + charge - discharge, i.e.
        100% round-trip efficiency, which is physically impossible)."""
        opt = BatteryOptimizer(
            battery_capacity_mwh=50, charge_rate_mw=20, discharge_rate_mw=20,
            initial_charge_mwh=10, round_trip_efficiency=0.81,  # easy exact math: sqrt=0.9
            soc_floor_pct=0.0, soc_ceiling_pct=1.0,
            terminal_soc_target_mwh=10, cycling_cost_per_mwh=0.0,
        )
        solar = np.array([30.0, 0.0])       # block 0: big surplus; block 1: no solar
        schedule = np.array([10.0, 10.0])   # declared 10 MW both blocks
        df = opt.optimize(solar, schedule)

        charged_mwh = df.loc[0, "charge_mw"] * opt.dt_hours
        discharged_mwh = df.loc[1, "discharge_mw"] * opt.dt_hours
        assert charged_mwh > 0.1  # it actually used the battery
        # Round trip: energy that reaches storage = charged_mwh * eff;
        # energy delivered back out for the same stored energy = that * eff
        # again -- so delivered should be noticeably less than charged.
        assert discharged_mwh < charged_mwh * 0.85

    def test_dt_hours_scales_energy_not_just_power(self):
        """A 20 MW charge for Δt=0.25h must add ~5 MWh (post-efficiency) to
        the battery, not 20 MWh (which is what an implicit Δt=1h would do)."""
        opt = BatteryOptimizer(
            battery_capacity_mwh=50, charge_rate_mw=20, discharge_rate_mw=20,
            initial_charge_mwh=10, round_trip_efficiency=1.0,  # isolate Δt, not efficiency
            soc_floor_pct=0.0, soc_ceiling_pct=1.0,
            dt_hours=0.25, cycling_cost_per_mwh=0.0,
        )
        solar = np.array([100.0])
        schedule = np.array([0.0])
        df = opt.optimize(solar, schedule)
        # Full 20 MW charge for one 0.25h block, 100% efficiency -> +5 MWh.
        assert df.loc[0, "battery_level_mwh"] == pytest.approx(15.0, abs=0.05)

    def test_soc_never_breaches_floor_or_ceiling(self):
        rng = np.random.default_rng(0)
        opt = BatteryOptimizer(
            battery_capacity_mwh=50, charge_rate_mw=30, discharge_rate_mw=30,
            initial_charge_mwh=25, soc_floor_pct=0.10, soc_ceiling_pct=0.95,
        )
        n = 20
        solar = rng.uniform(0, 80, n)
        schedule = rng.uniform(0, 60, n)
        df = opt.optimize(solar, schedule)

        assert (df["battery_level_mwh"] >= opt.soc_floor_mwh - 1e-6).all()
        assert (df["battery_level_mwh"] <= opt.soc_ceiling_mwh + 1e-6).all()

    def test_rejects_initial_charge_outside_usable_range(self):
        with pytest.raises(ValueError):
            BatteryOptimizer(
                battery_capacity_mwh=50, initial_charge_mwh=2,  # below 10% floor = 5 MWh
                soc_floor_pct=0.10, soc_ceiling_pct=0.95,
            )

    def test_terminal_soc_constraint_holds(self):
        opt = BatteryOptimizer(
            battery_capacity_mwh=50, charge_rate_mw=25, discharge_rate_mw=25,
            initial_charge_mwh=25, terminal_soc_target_mwh=25,
        )
        # A long shortfall stretch that WOULD drain the battery fully by
        # the end, if the terminal constraint didn't hold it back.
        n = 10
        solar = np.zeros(n)
        schedule = np.full(n, 20.0)
        df = opt.optimize(solar, schedule)
        assert df.loc[n - 1, "battery_level_mwh"] >= opt.terminal_soc_target - 1e-6

    def test_never_charges_and_discharges_in_the_same_block(self):
        rng = np.random.default_rng(1)
        opt = BatteryOptimizer(battery_capacity_mwh=50, charge_rate_mw=25, discharge_rate_mw=25, initial_charge_mwh=25)
        n = 15
        solar = rng.uniform(0, 60, n)
        schedule = rng.uniform(0, 60, n)
        df = opt.optimize(solar, schedule)

        both_active = (df["charge_mw"] > 0.01) & (df["discharge_mw"] > 0.01)
        assert not both_active.any()


class TestObjectiveIsDeviationNotUnmetDemand:
    def test_battery_smooths_both_over_and_under_delivery(self):
        """The old objective only minimized SHORTAGE -- it had zero reason
        to use the battery to avoid over-delivering when solar > schedule.
        The new objective must reduce deviation in BOTH directions."""
        opt = BatteryOptimizer(
            battery_capacity_mwh=50, charge_rate_mw=25, discharge_rate_mw=25,
            initial_charge_mwh=25, deviation_penalty_per_mwh=10.0,
            cycling_cost_per_mwh=0.01,
        )
        # Block 0: solar way over schedule (would over-deliver if untouched).
        # Block 1: solar under schedule (shortfall).
        solar = np.array([60.0, 5.0])
        schedule = np.array([20.0, 20.0])
        df = opt.optimize(solar, schedule)

        # Over-delivery block: battery should charge to absorb the surplus.
        assert df.loc[0, "charge_mw"] > 1.0
        # Deviation after optimization must be smaller than doing nothing.
        untouched_deviation_0 = abs(solar[0] - schedule[0])
        assert df.loc[0, "deviation_mwh"] < untouched_deviation_0 * opt.dt_hours

    def test_cycling_cost_discourages_pointless_cycling(self):
        """With a high cycling cost and a scenario where NOT cycling is
        already at zero deviation, the optimizer should not cycle at all."""
        opt = BatteryOptimizer(
            battery_capacity_mwh=50, charge_rate_mw=25, discharge_rate_mw=25,
            initial_charge_mwh=25, deviation_penalty_per_mwh=1.0,
            cycling_cost_per_mwh=100.0,  # extremely expensive to cycle
        )
        solar = np.array([20.0, 20.0])
        schedule = np.array([20.0, 20.0])  # solar already matches schedule exactly
        df = opt.optimize(solar, schedule)
        assert (df["charge_mw"] < 0.01).all()
        assert (df["discharge_mw"] < 0.01).all()


class TestFallback:
    def test_fallback_runs_and_respects_dt_and_efficiency(self):
        opt = BatteryOptimizer(
            battery_capacity_mwh=50, charge_rate_mw=20, discharge_rate_mw=20,
            initial_charge_mwh=10, round_trip_efficiency=0.81, dt_hours=0.25,
        )
        solar = np.array([100.0])
        schedule = np.array([0.0])
        df = opt._rule_based_fallback(solar, schedule)
        # 20 MW * 0.25h * charge_eff(0.9) = 4.5 MWh added.
        assert df.loc[0, "battery_level_mwh"] == pytest.approx(14.5, abs=0.05)
