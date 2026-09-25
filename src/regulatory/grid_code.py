"""
grid_code.py — CERC Indian Electricity Grid Code (IEGC) 2023 scheduling
and revision-timing rules (roadmap P1.4).

SOURCE (read directly, not summarized from memory):
  - CERC (Indian Electricity Grid Code) Regulations, 2023, full text,
    published 11.07.2023 (Gazette Part-III, Section-4, No. 488); brought
    into force from 01.10.2023 by CERC notification No. L-1/265/2022/CERC
    dated 03.08.2023 (Regulation 1(2) left the date to a separate
    notification). Regulation 49(1) (the complete D-1 scheduling
    timeline, including the Real-Time Market procedure at 49(1)(q)/(r))
    and Regulation 49(4)/(7)/(8)/(9) (revision eligibility and timing)
    read directly from the regulation text itself, not a secondary
    summary.
  - Order in Petition No. 14/SM/2023 ("Removal of difficulties, First
    Order"), dated 30.09.2023 -- quotes and clarifies Regulation 49(4)(c),
    49(7), 49(8), 49(9), 46, 47, and gives a worked numerical example this
    module's revision_effective_timestamp() is checked against.
  - Order in Petition No. 18/SM/2023 ("Removal of difficulties, Second
    Order"), dated 18.12.2023 -- quotes Regulation 49(1)(f)(i), 49(4)(b)
    (ii)/(c), 46(4)(c)/(d), and further modifies the temporary Reg 49(7)
    revision-count dispensation (not relevant to WS sellers -- see below).

WHAT THIS COVERS:
  Two real, cited facts about a WS (wind/solar) seller's schedule:

  1. REVISION ELIGIBILITY (Regulation 49(8)): "In case of requirement of
     revision of schedule due to forecasting error, a WS seller may
     revise its schedule only in case of bilateral transactions and not
     in case of collective transaction." A plant selling purely through
     collective (power-exchange/pooled) transactions cannot revise its
     schedule at all under this clause -- only a plant with a bilateral
     PPA structure can. `can_revise_schedule` encodes exactly this gate.

  2. REVISION-EFFECTIVE TIMING (Regulation 49(4)(c), cross-referenced by
     49(8)): "any revision in schedule made in odd time blocks shall
     become effective from 7th time block and any revision in schedule
     made in even time blocks shall become effective from 8th time
     block, counting the time block in which the request for revision
     has been received... to be the first one." This is Indian grid
     scheduling's real "gate closure" -- a revision is never instant; it
     always takes 6 more full blocks (if requested in an odd block) or 7
     more full blocks (if requested in an even block) beyond the block
     the request itself falls in, i.e. roughly 90-105 minutes of
     unavoidable lead time on top of however much of the current block
     has already elapsed.

     `revision_effective_timestamp` is verified against the CERC order's
     OWN worked numerical example (14/SM/2023, para 14): a request at
     1:56 PM (which falls in block 56, 13:45-14:00 -- an EVEN block)
     becomes effective at block 63, i.e. 15:30-15:45 ("3.30-3.45 pm",
     exactly what the order itself states) -- 56 + 7 = 63, matching the
     "8th time block, counting 56 as the first" rule exactly.

  3. REAL-TIME MARKET (RTM) GATE CLOSURE (Regulation 49(1)(q)/(r)): the
     mechanism most solar+battery IPPs would actually use to correct
     their position close to real time, distinct from the bilateral
     Regulation 49(8) revision above. RTM trades in half-hour delivery
     windows: the bidding window for a given half-hour opens 75 minutes
     before it starts and CLOSES (gate closure -- no more bids) 60
     minutes before it starts; power exchanges then clear bids in the
     15 minutes after gate closure, and RLDC publishes the final schedule
     5 minutes before delivery begins. The regulation text gives one
     concrete instance directly: "window for trade in real-time market
     for day (D) shall open from 22.45 hrs to 23.00 hrs of (D-1) for the
     delivery of power for the first two time-blocks ... 00.00 hrs to
     00.30 hrs, and will be repeated every half an hour thereafter" --
     i.e. every half-hour window follows the same 75/60-minutes-before
     pattern. `rtm_gate_closure_timestamp` and `rtm_delivery_window_start`
     implement this directly, verified against that exact instance.

WHAT THIS DOES NOT COVER:
  - A numeric cap on how many times per day a WS seller may revise under
    Regulation 49(8) (the bilateral path, not RTM). The documents read
    place an explicit numeric cap (2, then up to 4-6/day) on Regulation
    49(7) revisions (forced outage / partial outage of general sellers --
    thermal, hydro, gas), NOT on Regulation 49(8)'s WS forecasting-error
    revisions. Rather than invent a number for WS sellers, this module
    places no cap; if CERC later specifies one for WS sellers
    specifically, add it here.
  - Actually submitting bids into RTM or a real day-ahead schedule to any
    real Load Despatch Centre or Power Exchange -- this module only
    computes the TIMING such submissions would be subject to.
"""

import pandas as pd

from src.time_blocks import BLOCK_MINUTES, block_of

# Regulation 49(8): WS seller schedule revision is allowed only for
# bilateral transactions, not collective (exchange/pooled) transactions.
WS_SELLER_REVISION_ELIGIBLE_TRANSACTION_TYPES = frozenset({"bilateral"})

# The two real-world WS-seller transaction structures Regulation 49(8)
# distinguishes. Only "bilateral" may revise a schedule at all (see
# above) -- this is the full set of values plant onboarding (roadmap
# P2.3) accepts for regulatory.transaction_type, so an onboarding caller
# who types something else is told the two real options, not a bare
# "invalid value".
KNOWN_TRANSACTION_TYPES = frozenset({"bilateral", "collective"})

# The complete D-1 day-ahead scheduling timeline, Regulation 49(1) --
# reference/display only; nothing in this codebase submits a real D-1
# schedule to a Load Despatch Centre or Power Exchange yet, so none of
# this is enforced anywhere.
DAY_AHEAD_TIMELINE = {
    "declared_capacity_submission_deadline": "06:00",   # Reg 49(1)(a) -- DC/available-capacity submission by generators
    "beneficiary_entitlement_declared_by": "07:00",     # Reg 49(1)(b)
    "cross_border_requisition_deadline": "08:00",       # Reg 49(1)(d)
    "buyer_requisition_deadline": "08:00",              # Reg 49(1)(f)(i)/(ii)
    "gna_corridor_allocation_intimated_by": "08:15",    # Reg 49(1)(g)(i)
    "gna_requisition_revision_deadline": "08:30",       # Reg 49(1)(g)(ii)
    "gna_final_schedules_issued_by": "09:00",           # Reg 49(1)(g)(iii)
    "tgna_requisition_deadline": "09:15",               # Reg 49(1)(j)(i)
    "section62_surplus_dam_sale": "09:45",              # Reg 49(1)(l)
    "tgna_final_schedules_issued_by": "09:45",          # Reg 49(1)(j)(iv)
    "dam_collective_bidding_window": ("10:00", "11:00"),  # Reg 49(1)(m)(i)
    "dam_final_trade_schedules_by": "13:00",            # Reg 49(1)(m)(iv)
    "exigency_tgna_processed_by": "14:00",              # Reg 49(1)(o)
    "scuc_candidate_list_after": "14:30",               # Reg 46(4)(c)
    "scuc_incremental_scheduling_by": "15:00",          # Reg 46(4)(d)
}

# Regulation 49(1)(q): RTM bid window length and gate closure, in minutes
# before the half-hour delivery window it covers starts.
RTM_DELIVERY_WINDOW_MINUTES = 30
RTM_BID_WINDOW_OPENS_BEFORE_DELIVERY_MINUTES = 75  # e.g. 22:45 for a 00:00 delivery window
RTM_GATE_CLOSURE_BEFORE_DELIVERY_MINUTES = 60      # e.g. 23:00 for a 00:00 delivery window


def can_revise_schedule(transaction_type: str) -> bool:
    """Regulation 49(8): only a WS seller with a BILATERAL transaction
    structure may revise its schedule for forecasting error at all."""
    return transaction_type in WS_SELLER_REVISION_ELIGIBLE_TRANSACTION_TYPES


def revision_effective_timestamp(request_timestamp, tz: str = "Asia/Kolkata") -> pd.Timestamp:
    """Regulation 49(4)(c): the timestamp at which a schedule revision
    requested at `request_timestamp` actually takes effect.

    Verified against 14/SM/2023's own worked example (para 14): a
    request at 1:56 PM falls in block 56 (13:45-14:00, EVEN) and becomes
    effective at block 63 (15:30-15:45) -- exactly 56 + 7.
    """
    request_timestamp = pd.Timestamp(request_timestamp)
    block = block_of(request_timestamp)  # 1-96, within request's own calendar day
    lead_blocks = 6 if block % 2 == 1 else 7  # odd -> 7th block (k+6); even -> 8th block (k+7)

    day_start = request_timestamp.normalize()
    block_start = day_start + pd.Timedelta(minutes=(block - 1) * BLOCK_MINUTES)
    return block_start + pd.Timedelta(minutes=lead_blocks * BLOCK_MINUTES)


def rtm_delivery_window_start(timestamp) -> pd.Timestamp:
    """The start of the half-hour RTM delivery window containing
    `timestamp` (Regulation 49(1)(q): windows are aligned to the hour and
    half-hour -- 00:00-00:30, 00:30-01:00, and so on)."""
    timestamp = pd.Timestamp(timestamp)
    minutes_since_midnight = (timestamp - timestamp.normalize()).total_seconds() / 60
    window_index = int(minutes_since_midnight // RTM_DELIVERY_WINDOW_MINUTES)
    return timestamp.normalize() + pd.Timedelta(minutes=window_index * RTM_DELIVERY_WINDOW_MINUTES)


def rtm_gate_closure_timestamp(delivery_timestamp) -> pd.Timestamp:
    """Regulation 49(1)(q): the RTM bid gate-closure instant for the
    half-hour delivery window containing `delivery_timestamp` -- 60
    minutes before that window starts. Verified against the regulation's
    own concrete instance: for the 00:00-00:30 delivery window, gate
    closure is 23:00 hrs of the previous day (00:00 - 60min)."""
    return rtm_delivery_window_start(delivery_timestamp) - pd.Timedelta(minutes=RTM_GATE_CLOSURE_BEFORE_DELIVERY_MINUTES)


def rtm_bid_window_open_timestamp(delivery_timestamp) -> pd.Timestamp:
    """Regulation 49(1)(q): when RTM bidding OPENS for the half-hour
    delivery window containing `delivery_timestamp` -- 75 minutes before
    that window starts. Verified against the regulation's own concrete
    instance: for the 00:00-00:30 delivery window, bidding opens at 22:45
    hrs of the previous day (00:00 - 75min)."""
    return rtm_delivery_window_start(delivery_timestamp) - pd.Timedelta(minutes=RTM_BID_WINDOW_OPENS_BEFORE_DELIVERY_MINUTES)
