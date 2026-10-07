"""
engine.py — Operator recommendation rules (roadmap P2.6).

WHY THIS EXISTS:
  Turns the already-computed P2.4 (loss attribution) and P2.5
  (schedule-risk) evidence into a short list of concrete suggestions for
  a human operator. Per the roadmap: "human-approved only ... never
  automatic control." Nothing in this module writes to a schedule, a
  battery, or anything else -- it only proposes, and every proposal
  names the rule that fired (`trigger`) and the numbers behind it
  (`evidence`), so an operator can verify it before approving.
  Persisting the resulting approve/dismiss decision is
  src.recommendations.store's job, not this module's -- this module is
  pure (same inputs -> same recommendations, no I/O).
"""

import pandas as pd

from src.regulatory.grid_code import can_revise_schedule, revision_effective_timestamp

TYPE_SCHEDULE_REVISION = "schedule_revision"
TYPE_BATTERY_ACTION = "battery_action"
TYPE_INSPECTION = "inspection"

RISK_HIGH = "high"
RISK_MEDIUM = "medium"


def _recommendation(recommendation_type: str, trigger: str, evidence: dict,
                     suggested_action: str, timestamp=None) -> dict:
    return {
        "recommendation_type": recommendation_type,
        "trigger": trigger,
        "evidence": evidence,
        "suggested_action": suggested_action,
        "timestamp": timestamp.isoformat() if hasattr(timestamp, "isoformat") else timestamp,
    }


def recommend_schedule_revisions(risk_report, plant_config: dict, now=None) -> list:
    """
    One recommendation per HIGH-risk block (src.risk.schedule_risk,
    roadmap P2.5) whose revision gate is still open.

    Skipped entirely (not just "gate closed") when the plant's own
    configured transaction structure isn't revision-eligible at all
    (CERC IEGC 2023 Regulation 49(8): only "bilateral" may revise) --
    see src/regulatory/grid_code.py. Skipped for one specific block when
    the gate for THAT block has already closed (Regulation 49(4)(c)'s
    7th/8th time-block rule, src.regulatory.grid_code.
    revision_effective_timestamp) -- a battery action may still apply to
    that block; see `recommend_battery_actions`.

    When a block carries no real timestamp, or `now` isn't given, the
    gate-closure timing can't be checked at all -- the recommendation is
    still produced, but its evidence says so explicitly rather than
    silently assuming the gate is open.
    """
    revision_windows = (plant_config.get("regulatory") or {}).get("revision_windows") or {}
    transaction_type = revision_windows.get("transaction_type")

    if transaction_type is not None and not can_revise_schedule(transaction_type):
        return []  # structurally ineligible to revise at all -- nothing to recommend here

    recommendations = []
    for block in risk_report.blocks:
        if block.risk_level != RISK_HIGH:
            continue

        gate_checkable = block.timestamp is not None and now is not None
        if gate_checkable:
            effective_ts = revision_effective_timestamp(now)
            block_ts = pd.Timestamp(block.timestamp)
            if block_ts < effective_ts:
                continue  # this specific block's gate has already closed
            gate_note = (
                f"a revision requested now would take effect at {effective_ts.isoformat()}, "
                f"on or before this block's own {block_ts.isoformat()} -- still revisable."
            )
        else:
            gate_note = "gate-closure timing not checked -- no real block timestamp/now given."

        recommendations.append(_recommendation(
            TYPE_SCHEDULE_REVISION,
            trigger="schedule_risk_high",
            evidence={
                "worst_band": block.worst_band,
                "quantile_bands": block.quantile_bands,
                "quantile_net_rs": block.quantile_net_rs,
                "gate_note": gate_note,
            },
            suggested_action=(
                f"Revise the declared schedule for this block toward the P50 forecast "
                f"({block.p50_mw:.1f} MW) -- currently declared at {block.declared_schedule_mw:.1f} MW, "
                f"which reaches DSM band {block.worst_band} in the worst case."
            ),
            timestamp=block.timestamp,
        ))
    return recommendations


def recommend_battery_actions(risk_report, battery_state: dict, dt_hours: float) -> list:
    """
    One recommendation per MEDIUM/HIGH-risk block where the battery has
    real headroom to reduce the P50 deviation, using whichever direction
    (charge to absorb over-injection, discharge to cover under-injection)
    the median forecast calls for.

    `battery_state`: {"soc_mwh", "capacity_mwh", "charge_rate_mw",
    "discharge_rate_mw"}. Pass None to skip entirely (no battery
    configured for this plant) -- this rule never assumes battery specs
    that weren't given.
    """
    if not battery_state:
        return []

    recommendations = []
    for block in risk_report.blocks:
        if block.risk_level not in (RISK_MEDIUM, RISK_HIGH):
            continue

        deviation_mwh = block.quantile_deviation_mwh["p50"]
        if abs(deviation_mwh) < 1e-9:
            continue  # median forecast agrees with the declared schedule -- nothing to offset

        if deviation_mwh > 0:
            direction = "charge"
            headroom_mwh = min(
                battery_state["capacity_mwh"] - battery_state["soc_mwh"],
                battery_state["charge_rate_mw"] * dt_hours,
            )
        else:
            direction = "discharge"
            headroom_mwh = min(battery_state["soc_mwh"], battery_state["discharge_rate_mw"] * dt_hours)

        headroom_mwh = max(headroom_mwh, 0.0)
        if headroom_mwh <= 1e-9:
            continue  # no usable headroom in the needed direction -- nothing to recommend

        needed_mwh = abs(deviation_mwh)
        achievable_mwh = min(needed_mwh, headroom_mwh)
        is_partial = achievable_mwh < needed_mwh - 1e-9

        recommendations.append(_recommendation(
            TYPE_BATTERY_ACTION,
            trigger=f"schedule_risk_{block.risk_level}",
            evidence={
                "direction": direction,
                "p50_deviation_mwh": round(deviation_mwh, 4),
                "battery_headroom_mwh": round(headroom_mwh, 4),
                "achievable_mwh": round(achievable_mwh, 4),
                "partial": is_partial,
            },
            suggested_action=(
                f"{'Charge' if direction == 'charge' else 'Discharge'} the battery by up to "
                f"{achievable_mwh:.2f} MWh to offset the P50 forecast's "
                f"{abs(deviation_mwh):.2f} MWh {'over' if direction == 'charge' else 'under'}-injection "
                f"against the declared schedule"
                + (f" (only covers {achievable_mwh:.2f} of {needed_mwh:.2f} MWh -- battery headroom "
                   f"is insufficient to fully offset this block)" if is_partial else ".")
            ),
            timestamp=block.timestamp,
        ))
    return recommendations


def recommend_inspections(loss_report) -> list:
    """
    One recommendation per SUSPECTED-EQUIPMENT run (src.attribution.loss,
    roadmap P2.4) -- grouped by that run's own start timestamp (set by
    attribute_losses itself) so one physical event produces one
    recommendation, not one per 15-minute block inside it.
    """
    equipment_blocks = [b for b in loss_report.blocks if b.cause == "equipment"]
    if not equipment_blocks:
        return []

    runs = {}
    for block in equipment_blocks:
        run_start = block.evidence.get("run_start_timestamp")
        runs.setdefault(run_start, []).append(block)

    recommendations = []
    for run_start, blocks in runs.items():
        blocks.sort(key=lambda b: b.timestamp)
        total_loss_mw = sum(b.loss_mw for b in blocks)
        recommendations.append(_recommendation(
            TYPE_INSPECTION,
            trigger="suspected_equipment_run",
            evidence={
                "run_start_timestamp": run_start,
                "run_length_blocks": blocks[0].evidence.get("run_length_blocks"),
                "total_loss_mw_summed_across_run": round(total_loss_mw, 3),
                "block_timestamps": [
                    b.timestamp.isoformat() if hasattr(b.timestamp, "isoformat") else str(b.timestamp)
                    for b in blocks
                ],
            },
            suggested_action=(
                f"Schedule a physical inspection -- output stayed below expected for "
                f"{len(blocks)} consecutive blocks starting {run_start}, not explained by weather, "
                f"consistent with a hardware fault. This platform has no per-inverter telemetry, so "
                f"the affected asset is not identified -- an on-site check is needed to confirm."
            ),
            timestamp=blocks[0].timestamp,
        ))
    return recommendations
