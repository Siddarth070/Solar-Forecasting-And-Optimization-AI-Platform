"""
grid_code.py — CERC Indian Electricity Grid Code (IEGC) 2023 scheduling
and revision-timing rules (roadmap P1.4).

SOURCE (read directly, not summarized from memory):
  - CERC (Indian Electricity Grid Code) Regulations, 2023, published
    11.07.2023 (Gazette Part-III, Section-4, No. 488); brought into force
    from 01.10.2023 by CERC notification No. L-1/265/2022/CERC dated
    03.08.2023 (Regulation 1(2) left the date to a separate notification).
  - Order in Petition No. 14/SM/2023 ("Removal of difficulties, First
    Order"), dated 30.09.2023 -- quotes and clarifies Regulation 49(4)(c),
    49(7), 49(8), 49(9), 46, 47.
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

WHAT THIS DOES NOT COVER:
  - A numeric cap on how many times per day a WS seller may revise. The
    documents read place an explicit numeric cap (2, then up to 4-6/day)
    on Regulation 49(7) revisions (forced outage / partial outage of
    general sellers -- thermal, hydro, gas), NOT on Regulation 49(8)'s WS
    forecasting-error revisions. Rather than invent a number for WS
    sellers, this module places no cap; if CERC later specifies one for
    WS sellers specifically, add it here.
  - Day-ahead scheduling deadlines (buyer requisition by 8 AM D-1 per
    Reg 49(1)(f)(i), un-requisitioned Section-62 surplus sale "as
    available at 9.45 AM" per Reg 49(1)(l), SCUC list preparation after
    1430 hrs and incremental scheduling by 1500 hrs D-1 per Reg 46(4)
    (c)/(d)) are recorded in DAY_AHEAD_TIMELINE below for reference and
    display, but this platform does not yet submit real D-1 schedules to
    any real Load Despatch Centre, so nothing enforces them.
"""

import pandas as pd

from src.time_blocks import BLOCK_MINUTES, block_of

# Regulation 49(8): WS seller schedule revision is allowed only for
# bilateral transactions, not collective (exchange/pooled) transactions.
WS_SELLER_REVISION_ELIGIBLE_TRANSACTION_TYPES = frozenset({"bilateral"})

# Reference only (see WHAT THIS DOES NOT COVER above) -- not enforced
# anywhere in this codebase, since nothing here submits real D-1
# schedules to a Load Despatch Centre yet.
DAY_AHEAD_TIMELINE = {
    "buyer_requisition_deadline": "08:00",       # Reg 49(1)(f)(i)
    "gna_transactions_scheduled_by": "09:00",    # 14/SM/2023 order, para 41
    "section62_surplus_dam_sale": "09:45",       # Reg 49(1)(l)
    "exigency_tgna_scheduled_after": "13:00",    # 14/SM/2023 order, para 41
    "scuc_candidate_list_after": "14:30",        # Reg 46(4)(c)
    "scuc_incremental_scheduling_by": "15:00",   # Reg 46(4)(d)
}


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
