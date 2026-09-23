"""
test_recommendations_store.py
--------------------------------
Roadmap P2.6 acceptance criterion: "an explicit approve/dismiss action
that is logged." Exercises the SQLite-backed append-only log directly
(src/recommendations/store.py) -- every test uses an in-memory DB so
these run fast with no filesystem side effects.

RUN WITH:
  pytest tests/test_recommendations_store.py -v
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.recommendations.store import (
    connect,
    decide_recommendation,
    get_recommendation,
    list_recommendations,
    log_recommendation,
)


@pytest.fixture
def conn():
    return connect(":memory:")


class TestLoggingAndReadingBack:
    def test_a_freshly_logged_recommendation_is_pending(self, conn):
        rec_id = log_recommendation(
            conn, plant_id="jaipur_100mw", recommendation_type="inspection",
            trigger="suspected_equipment_run", evidence={"run_length_blocks": 4},
            suggested_action="Schedule a physical inspection.",
        )
        rec = get_recommendation(conn, rec_id)
        assert rec["status"] == "pending"
        assert rec["decided_by"] is None
        assert rec["evidence"] == {"run_length_blocks": 4}
        assert rec["plant_id"] == "jaipur_100mw"

    def test_unknown_id_returns_none(self, conn):
        assert get_recommendation(conn, 9999) is None


class TestDecisions:
    def _log(self, conn):
        return log_recommendation(
            conn, plant_id="jaipur_100mw", recommendation_type="battery_action",
            trigger="schedule_risk_high", evidence={"direction": "discharge"},
            suggested_action="Discharge the battery.",
        )

    def test_approving_updates_status_and_records_who(self, conn):
        rec_id = self._log(conn)
        result = decide_recommendation(conn, rec_id, "approved", decided_by="ops_alice", note="looks right")
        assert result["status"] == "approved"
        assert result["decided_by"] == "ops_alice"
        assert result["decision_note"] == "looks right"
        assert result["decided_at"] is not None

    def test_dismissing_is_also_logged(self, conn):
        rec_id = self._log(conn)
        result = decide_recommendation(conn, rec_id, "dismissed", decided_by="ops_bob")
        assert result["status"] == "dismissed"
        assert result["decided_by"] == "ops_bob"

    def test_invalid_decision_value_raises(self, conn):
        rec_id = self._log(conn)
        with pytest.raises(ValueError):
            decide_recommendation(conn, rec_id, "maybe", decided_by="ops_alice")

    def test_missing_decided_by_raises(self, conn):
        rec_id = self._log(conn)
        with pytest.raises(ValueError):
            decide_recommendation(conn, rec_id, "approved", decided_by="")

    def test_deciding_a_nonexistent_recommendation_raises_keyerror(self, conn):
        with pytest.raises(KeyError):
            decide_recommendation(conn, 9999, "approved", decided_by="ops_alice")

    def test_a_later_decision_overrides_status_without_losing_the_earlier_one(self, conn):
        rec_id = self._log(conn)
        decide_recommendation(conn, rec_id, "approved", decided_by="ops_alice")
        result = decide_recommendation(conn, rec_id, "dismissed", decided_by="ops_bob", note="reversed")
        assert result["status"] == "dismissed"
        assert result["decided_by"] == "ops_bob"
        # the earlier decision is still in the append-only log, not overwritten
        history = conn.execute(
            "SELECT decision, decided_by FROM recommendation_decisions WHERE recommendation_id = ? ORDER BY id",
            (rec_id,),
        ).fetchall()
        assert [(h["decision"], h["decided_by"]) for h in history] == [
            ("approved", "ops_alice"), ("dismissed", "ops_bob"),
        ]


class TestListing:
    def test_filters_by_plant_id_and_status(self, conn):
        id_a = log_recommendation(conn, "jaipur_100mw", "inspection", "t", {}, "action a")
        id_b = log_recommendation(conn, "pune_50mw", "inspection", "t", {}, "action b")
        decide_recommendation(conn, id_a, "approved", decided_by="ops_alice")

        jaipur_only = list_recommendations(conn, plant_id="jaipur_100mw")
        assert [r["id"] for r in jaipur_only] == [id_a]

        pending_only = list_recommendations(conn, status="pending")
        assert [r["id"] for r in pending_only] == [id_b]

        approved_only = list_recommendations(conn, status="approved")
        assert [r["id"] for r in approved_only] == [id_a]

    def test_newest_first(self, conn):
        id_a = log_recommendation(conn, "jaipur_100mw", "inspection", "t", {}, "a")
        id_b = log_recommendation(conn, "jaipur_100mw", "inspection", "t", {}, "b")
        assert [r["id"] for r in list_recommendations(conn)] == [id_b, id_a]
