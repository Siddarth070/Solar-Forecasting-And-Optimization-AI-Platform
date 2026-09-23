"""
schedule_risk.py — Per-block DSM schedule-risk scoring (roadmap P2.5).

WHY THIS EXISTS:
  A P10/P50/P90 forecast (roadmap P1.3) is only useful to an operator if
  it's translated into "what could this cost me against my declared
  schedule". This module does exactly that translation for one 15-minute
  block at a time: it runs each of the three quantiles through the real
  CERC DSM settlement math (`src.regulatory.dsm.deviation_settlement`,
  built for roadmap P1.7) against the plant's own configured contract
  rate and seller category, and reports which Note-1 volume-limit band
  each quantile's deviation would land in. The spread across P10/P50/P90
  IS the risk: if even the pessimistic and optimistic ends of the
  forecast stay inside the cheapest band, there's little to worry about;
  if the bands disagree, or the median already reaches the worst band,
  that's a real, quantified exposure worth surfacing before the block's
  gate closes (roadmap P1.4's `src.regulatory.grid_code`).

WHAT THIS IS NOT:
  Not a settlement statement. The real Rupee number a plant is billed
  depends on the actual metered injection (not a forecast), the real
  contract rate (not the illustrative placeholder in
  configs/plants/*.yaml -- see README Known Gaps), and CERC's own
  post-01.04.2026 blend-weight order once it exists (see
  src/regulatory/dsm.py's own documented gap). Per the roadmap, every
  output below carries `DISCLAIMER` verbatim for exactly this reason.
"""

import datetime as _dt
from dataclasses import dataclass, field

from src.regulatory import dsm

DISCLAIMER = (
    "Indicative DSM exposure based on configured assumptions and uploaded data. "
    "Not an official settlement statement."
)

RISK_LOW = "low"
RISK_MEDIUM = "medium"
RISK_HIGH = "high"

QUANTILE_NAMES = ("p10", "p50", "p90")


def _band_touched(settlement: dict) -> int:
    """Which Note-1 volume-limit band a settlement's deviation reached:
    0 = no deviation, 1/2/3 = VLwS(1)/between/beyond VLwS(2) -- read
    directly off deviation_settlement's own segment widths rather than
    re-deriving the band cutoffs, so this can never disagree with the
    settlement Rs figure it's reporting alongside."""
    seg1, seg2, seg3 = settlement["segment_mwh"]
    if seg3 > 1e-9:
        return 3
    if seg2 > 1e-9:
        return 2
    if seg1 > 1e-9:
        return 1
    return 0


@dataclass
class ScheduleRiskBlock:
    timestamp: object
    declared_schedule_mw: float
    p10_mw: float
    p50_mw: float
    p90_mw: float
    risk_level: str
    worst_band: int
    quantile_bands: dict
    quantile_deviation_mwh: dict
    quantile_net_rs: dict
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat() if hasattr(self.timestamp, "isoformat") else self.timestamp,
            "declared_schedule_mw": round(self.declared_schedule_mw, 3),
            "p10_mw": round(self.p10_mw, 3),
            "p50_mw": round(self.p50_mw, 3),
            "p90_mw": round(self.p90_mw, 3),
            "risk_level": self.risk_level,
            "worst_band": self.worst_band,
            "quantile_bands": self.quantile_bands,
            "quantile_deviation_mwh": {k: round(v, 4) for k, v in self.quantile_deviation_mwh.items()},
            "quantile_net_rs": {k: round(v, 2) for k, v in self.quantile_net_rs.items()},
            "evidence": self.evidence,
            "disclaimer": DISCLAIMER,
        }


def _regulatory_inputs(plant_config: dict) -> tuple[str, float]:
    regulatory = plant_config.get("regulatory") or {}
    category = regulatory.get("seller_category")
    contract_rate = regulatory.get("contract_rate_rs_per_kwh")
    if category is None or contract_rate is None:
        raise ValueError(
            f"plant {plant_config.get('plant_id', 'unknown')!r} has no "
            f"regulatory.seller_category / regulatory.contract_rate_rs_per_kwh configured -- "
            f"schedule-risk scoring needs both to run the real DSM settlement math "
            f"(src.regulatory.dsm), unlike /optimize which can fall back to a flat penalty."
        )
    return category, contract_rate


def score_block(
    declared_schedule_mw: float,
    p10_mw: float,
    p50_mw: float,
    p90_mw: float,
    dt_hours: float,
    plant_config: dict,
    as_of: _dt.date,
    timestamp=None,
) -> ScheduleRiskBlock:
    """Score one 15-minute block's DSM exposure risk from its declared
    schedule and P10/P50/P90 forecast."""
    category, contract_rate = _regulatory_inputs(plant_config)
    capacity_mw = plant_config["capacity"]["ac_capacity_mw"]

    available_capacity_mwh = capacity_mw * dt_hours
    scheduled_mwh = declared_schedule_mw * dt_hours

    quantile_mw = {"p10": p10_mw, "p50": p50_mw, "p90": p90_mw}
    bands, deviations_mwh, net_rs = {}, {}, {}
    for name, mw in quantile_mw.items():
        actual_mwh = mw * dt_hours
        deviation_mwh = actual_mwh - scheduled_mwh
        settlement = dsm.deviation_settlement(
            deviation_mwh, available_capacity_mwh, contract_rate, category, as_of
        )
        bands[name] = _band_touched(settlement)
        deviations_mwh[name] = deviation_mwh
        net_rs[name] = settlement["net_rs"]

    worst_band = max(bands.values())
    if worst_band >= 3:
        risk_level = RISK_HIGH
    elif worst_band == 2:
        risk_level = RISK_MEDIUM
    else:
        risk_level = RISK_LOW

    evidence = {
        "available_capacity_mwh": round(available_capacity_mwh, 4),
        "scheduled_mwh": round(scheduled_mwh, 4),
        "band_spread": worst_band - min(bands.values()),
        "reason": (
            f"worst-case Note-1 band across the P10/P50/P90 forecast is band {worst_band} "
            f"(0 = no deviation, 1/2/3 = within/between/beyond the configured volume limits) -- "
            f"P10 reaches band {bands['p10']}, P50 reaches band {bands['p50']}, "
            f"P90 reaches band {bands['p90']}."
        ),
    }

    return ScheduleRiskBlock(
        timestamp=timestamp,
        declared_schedule_mw=declared_schedule_mw,
        p10_mw=p10_mw,
        p50_mw=p50_mw,
        p90_mw=p90_mw,
        risk_level=risk_level,
        worst_band=worst_band,
        quantile_bands=bands,
        quantile_deviation_mwh=deviations_mwh,
        quantile_net_rs=net_rs,
        evidence=evidence,
    )


@dataclass
class ScheduleRiskReport:
    plant_id: str
    blocks: list

    def risk_summary(self) -> dict:
        summary = {RISK_LOW: 0, RISK_MEDIUM: 0, RISK_HIGH: 0}
        for b in self.blocks:
            summary[b.risk_level] += 1
        return summary

    def to_dict(self) -> dict:
        return {
            "plant_id": self.plant_id,
            "blocks": len(self.blocks),
            "risk_summary": self.risk_summary(),
            "schedule": [b.to_dict() for b in self.blocks],
            "disclaimer": DISCLAIMER,
        }


def score_schedule_risk(
    declared_schedule_mw: list,
    p10_mw: list,
    p50_mw: list,
    p90_mw: list,
    dt_hours: float,
    plant_config: dict,
    as_of: _dt.date,
    timestamps: list = None,
) -> ScheduleRiskReport:
    """Score a full schedule of blocks -- see `score_block` for one block
    at a time. `timestamps`, if given, must be the same length as the
    other four lists."""
    lengths = {len(declared_schedule_mw), len(p10_mw), len(p50_mw), len(p90_mw)}
    if len(lengths) != 1:
        raise ValueError(
            "declared_schedule_mw, p10_mw, p50_mw and p90_mw must all have the same length, got "
            f"{len(declared_schedule_mw)}, {len(p10_mw)}, {len(p50_mw)}, {len(p90_mw)}."
        )
    if timestamps is not None and len(timestamps) != len(declared_schedule_mw):
        raise ValueError("timestamps, when given, must be the same length as the other lists.")

    ts_list = timestamps if timestamps is not None else [None] * len(declared_schedule_mw)
    blocks = [
        score_block(d, p10, p50, p90, dt_hours, plant_config, as_of, ts)
        for d, p10, p50, p90, ts in zip(declared_schedule_mw, p10_mw, p50_mw, p90_mw, ts_list)
    ]
    return ScheduleRiskReport(plant_id=plant_config.get("plant_id", "unknown"), blocks=blocks)
