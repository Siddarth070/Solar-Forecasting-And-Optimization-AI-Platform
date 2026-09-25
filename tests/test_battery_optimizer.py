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

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.optimization.battery_optimizer import BatteryOptimizer
from src.regulatory import dsm


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


class TestDsmMode:
    """Roadmap P1.7: the real CERC Regulation 8(4) WS-seller tiered
    structure (src/regulatory/dsm.py), wired into the optimizer's
    objective instead of the flat deviation_penalty_per_mwh, activated
    only when dsm_contract_rate_rs_per_kwh AND dsm_available_capacity_mw
    are both supplied."""

    AS_OF = dt.date(2026, 9, 22)  # post-01.04.2026 -- tighter solar bands

    def _opt(self, **overrides):
        kwargs = dict(
            battery_capacity_mwh=1000, charge_rate_mw=200, discharge_rate_mw=200,
            initial_charge_mwh=500, dt_hours=0.25, cycling_cost_per_mwh=0.0,
            round_trip_efficiency=1.0, soc_floor_pct=0.0, soc_ceiling_pct=1.0,
            terminal_soc_target_mwh=0,
            dsm_contract_rate_rs_per_kwh=2.5, dsm_available_capacity_mw=100,
            dsm_seller_category="solar", dsm_as_of=self.AS_OF,
        )
        kwargs.update(overrides)
        return BatteryOptimizer(**kwargs)

    def test_dsm_mode_is_off_by_default(self):
        opt = BatteryOptimizer(battery_capacity_mwh=50, charge_rate_mw=25,
                                discharge_rate_mw=25, initial_charge_mwh=25)
        assert opt.dsm_enabled is False

    def test_dsm_rs_column_absent_when_dsm_mode_off(self):
        opt = BatteryOptimizer(battery_capacity_mwh=50, charge_rate_mw=25,
                                discharge_rate_mw=25, initial_charge_mwh=25)
        df = opt.optimize(np.array([30.0, 20.0]), np.array([10.0, 10.0]))
        assert "dsm_rs" not in df.columns

    def test_reported_dsm_rs_matches_independent_recomputation(self):
        """The optimizer's reported dsm_rs for every block must match
        calling src/regulatory/dsm.py directly on that block's OWN
        deviation -- proves there's one source of truth for the charge,
        not a second, drifting copy of the regulation's math."""
        opt = self._opt(terminal_soc_target_mwh=500)
        rng = np.random.default_rng(5)
        solar = rng.uniform(0, 80, 10)
        schedule = rng.uniform(0, 60, 10)
        df = opt.optimize(solar, schedule)

        for _, row in df.iterrows():
            signed_deviation_mwh = (
                (row["solar_mw"] + row["discharge_mw"] - row["charge_mw"]) - row["declared_schedule_mw"]
            ) * opt.dt_hours
            expected = dsm.deviation_settlement(
                signed_deviation_mwh, opt.dsm_available_capacity_mwh, 2.5, "solar", self.AS_OF,
            )["net_rs"]
            # Reconstructed from the dataframe's own already-rounded (2dp)
            # MW columns, so a little slack accounts for rounding, not for
            # a real mismatch between the optimizer and src/regulatory/dsm.py.
            assert row["dsm_rs"] == pytest.approx(expected, abs=15.0)

    def test_dsm_mode_beats_doing_nothing_on_a_mixed_scenario(self):
        opt = self._opt()
        rng = np.random.default_rng(3)
        solar = rng.uniform(0, 80, 20)
        schedule = rng.uniform(0, 60, 20)
        df = opt.optimize(solar, schedule)

        baseline_rs = sum(
            dsm.deviation_settlement((s - sch) * opt.dt_hours, opt.dsm_available_capacity_mwh,
                                      2.5, "solar", self.AS_OF)["net_rs"]
            for s, sch in zip(solar, schedule)
        )
        assert df["dsm_rs"].sum() < baseline_rs

    def test_over_injection_revenue_saturates_beyond_vl2(self):
        """Regulation 8(4): over-injection beyond VLwS(2) is paid at ZERO.
        So once a block's surplus already exceeds VL2, injecting even
        MORE surplus must not change that block's settled dsm_rs at all
        -- a real, solver-tie-independent invariant since it's checked on
        the REPORTED cost, not on which exact MW split the LP picked."""
        opt = self._opt(charge_rate_mw=0.0, discharge_rate_mw=0.0)  # isolate: no battery action possible
        far_beyond_vl2 = opt.dsm_available_capacity_mw * 0.5  # available_capacity_mwh=25, VL2=2.5 MWh -> 10MW; 50MW is far beyond
        even_further = far_beyond_vl2 * 2

        df_a = opt.optimize(np.array([far_beyond_vl2]), np.array([0.0]))
        df_b = opt.optimize(np.array([even_further]), np.array([0.0]))
        assert df_a.loc[0, "dsm_rs"] == pytest.approx(df_b.loc[0, "dsm_rs"])

    def test_under_injection_within_vl1_costs_exactly_contract_rate(self):
        """A small, deliberately battery-unreachable shortfall (battery
        rates set to 0) must settle at exactly the contract rate -- the
        simplest, fully-deterministic case in Regulation 8(4)."""
        opt = self._opt(charge_rate_mw=0.0, discharge_rate_mw=0.0)
        # available_capacity_mwh = 100*0.25 = 25; VL1 (5%) = 1.25 MWh.
        # 4 MW * 0.25h = 1 MWh under-injection, safely within VL1.
        df = opt.optimize(np.array([16.0]), np.array([20.0]))
        assert df.loc[0, "dsm_rs"] == pytest.approx(1.0 * 2500 * 1.00)  # 2500 Rs


class TestExportLimit:
    """Roadmap P2.3: grid.export_limit_mw is a real, contractual cap on
    power delivered to the grid, which may be below the plant's own AC
    capacity. export_limit_mw=None (the default) must change nothing;
    when set, the LP must curtail (never just report an over-limit
    number) whenever solar generation alone exceeds the cap and the
    battery can't absorb the difference."""

    def test_no_export_limit_is_unchanged(self):
        # dt_hours=1.0 for simple arithmetic; identical inputs, only
        # export_limit_mw differs (omitted vs explicit None).
        kwargs = dict(
            battery_capacity_mwh=20, charge_rate_mw=5, discharge_rate_mw=5,
            initial_charge_mwh=10, dt_hours=1.0, round_trip_efficiency=1.0,
            soc_floor_pct=0.0, soc_ceiling_pct=1.0,
        )
        solar = np.array([30.0])
        schedule = np.array([10.0])
        df_default = BatteryOptimizer(**kwargs).optimize(solar, schedule)
        df_explicit_none = BatteryOptimizer(**kwargs, export_limit_mw=None).optimize(solar, schedule)
        assert df_default.equals(df_explicit_none)
        assert (df_default["curtailed_mw"] == 0.0).all()

    def test_export_limit_forces_curtailment_when_battery_cannot_absorb_surplus(self):
        # capacity=20 MWh, floor=0/ceiling=20 (pct 0/1), charge_rate=5,
        # dt=1h, eff=1 (round_trip_efficiency=1.0) -- simple arithmetic.
        # solar=50, schedule=10 -> surplus=40, but charge is capped at
        # charge_rate=5 regardless of surplus. export_limit=30 forces
        # curtailment of exactly 50 - 5(charge) - 30(cap) = 15 MW.
        opt = BatteryOptimizer(
            battery_capacity_mwh=20, charge_rate_mw=5, discharge_rate_mw=5,
            initial_charge_mwh=10, dt_hours=1.0, round_trip_efficiency=1.0,
            soc_floor_pct=0.0, soc_ceiling_pct=1.0, export_limit_mw=30,
        )
        df = opt.optimize(np.array([50.0]), np.array([10.0]))
        row = df.loc[0]
        assert row["charge_mw"] == pytest.approx(5.0, abs=0.01)
        assert row["curtailed_mw"] == pytest.approx(15.0, abs=0.01)
        grid_delivered = row["solar_mw"] - row["curtailed_mw"] + row["discharge_mw"] - row["charge_mw"]
        assert grid_delivered == pytest.approx(30.0, abs=0.01)
        assert grid_delivered <= 30.0 + 1e-6

    def test_export_limit_prefers_charging_over_curtailing_when_battery_has_headroom(self):
        # capacity=30 MWh, charge_rate=25, initial_charge=5 -> headroom
        # exactly 25 MWh, matching charge_rate. solar=40, schedule=10,
        # export_limit=15: charging the full 25 MW brings grid_delivered
        # to exactly 40-25=15=cap, so curtailment should be exactly zero
        # (charging costs less per MWh than curtailing, so the LP always
        # maxes charge first).
        opt = BatteryOptimizer(
            battery_capacity_mwh=30, charge_rate_mw=25, discharge_rate_mw=25,
            initial_charge_mwh=5, dt_hours=1.0, round_trip_efficiency=1.0,
            soc_floor_pct=0.0, soc_ceiling_pct=1.0, export_limit_mw=15,
            terminal_soc_target_mwh=5,
        )
        df = opt.optimize(np.array([40.0]), np.array([10.0]))
        row = df.loc[0]
        assert row["curtailed_mw"] == pytest.approx(0.0, abs=0.01)
        assert row["charge_mw"] == pytest.approx(25.0, abs=0.01)

    def test_rule_based_fallback_also_respects_export_limit(self):
        opt = BatteryOptimizer(
            battery_capacity_mwh=50, charge_rate_mw=20, discharge_rate_mw=20,
            initial_charge_mwh=10, round_trip_efficiency=0.81, dt_hours=0.25,
            export_limit_mw=10,
        )
        solar = np.array([100.0])
        schedule = np.array([0.0])
        df = opt._rule_based_fallback(solar, schedule)
        # charge_amt = 20 (rate-limited); grid_delivered = 100 - 20 = 80,
        # which exceeds export_limit=10 -> curtail the excess (70 MW),
        # leaving grid_delivered = 10 exactly. Battery level is untouched
        # by curtailment (matches the existing fallback test's own math).
        assert df.loc[0, "battery_level_mwh"] == pytest.approx(14.5, abs=0.05)
        assert df.loc[0, "curtailed_mw"] == pytest.approx(70.0, abs=0.01)
        assert df.loc[0, "grid_balance_mw"] == pytest.approx(10.0, abs=0.01)
