"""
store.py — Append-only operator-recommendation log (roadmap P2.6).

WHY SQLITE, NOT AN IN-MEMORY DICT:
  The roadmap requires "an explicit approve/dismiss action that is
  logged." An in-memory structure loses that log on every server
  restart, which isn't a log at all. This platform has no Postgres or
  other real database yet -- that's roadmap P2.11, a full architecture
  decision this session hasn't been asked to make. SQLite (Python's
  stdlib, one file, zero new infrastructure) is the smallest REAL,
  durable choice that actually satisfies "logged" without pretending to
  be more infrastructure than this project has.

SCHEMA:
  recommendations           -- one row per generated recommendation.
                                Immutable once written.
  recommendation_decisions  -- one row per approve/dismiss action,
                                append-only. A recommendation's CURRENT
                                status is its most recent decision (or
                                "pending" if none) -- so a human can
                                reverse an earlier call without the audit
                                trail losing what actually happened.
"""

import datetime as _dt
import json
import sqlite3
from pathlib import Path

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "recommendations.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS recommendations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plant_id TEXT NOT NULL,
    recommendation_type TEXT NOT NULL,
    trigger_name TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    suggested_action TEXT NOT NULL,
    block_timestamp TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS recommendation_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recommendation_id INTEGER NOT NULL REFERENCES recommendations(id),
    decision TEXT NOT NULL CHECK (decision IN ('approved', 'dismissed')),
    decided_by TEXT NOT NULL,
    decided_at TEXT NOT NULL,
    note TEXT DEFAULT ''
);
"""


def connect(db_path=DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open (creating if needed) the recommendation log. Pass ':memory:'
    for a throwaway DB, e.g. in tests.

    `check_same_thread=False`: FastAPI's TestClient (and its own worker
    threads in general) can call a request handler from a different
    thread than the one that opened a connection; this app never shares
    one connection across concurrent writers (each real request opens
    its own via `get_recommendations_db()`), so relaxing sqlite3's
    same-thread check is safe here, not a race condition being papered
    over."""
    if db_path != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def log_recommendation(
    conn: sqlite3.Connection,
    plant_id: str,
    recommendation_type: str,
    trigger: str,
    evidence: dict,
    suggested_action: str,
    block_timestamp=None,
) -> int:
    """Record a newly-generated recommendation. Returns its id."""
    created_at = _dt.datetime.now(_dt.timezone.utc).isoformat()
    ts = block_timestamp.isoformat() if hasattr(block_timestamp, "isoformat") else block_timestamp
    cur = conn.execute(
        "INSERT INTO recommendations "
        "(plant_id, recommendation_type, trigger_name, evidence_json, suggested_action, "
        "block_timestamp, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (plant_id, recommendation_type, trigger, json.dumps(evidence), suggested_action, ts, created_at),
    )
    conn.commit()
    return cur.lastrowid


def decide_recommendation(
    conn: sqlite3.Connection,
    recommendation_id: int,
    decision: str,
    decided_by: str,
    note: str = "",
) -> dict:
    """Log an explicit approve/dismiss action against a recommendation.
    Raises KeyError if the id doesn't exist."""
    if decision not in ("approved", "dismissed"):
        raise ValueError(f"decision must be 'approved' or 'dismissed', got {decision!r}")
    if not decided_by:
        raise ValueError("decided_by is required -- a decision must be attributable to someone.")

    exists = conn.execute("SELECT id FROM recommendations WHERE id = ?", (recommendation_id,)).fetchone()
    if exists is None:
        raise KeyError(f"no recommendation with id {recommendation_id}")

    decided_at = _dt.datetime.now(_dt.timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO recommendation_decisions (recommendation_id, decision, decided_by, decided_at, note) "
        "VALUES (?, ?, ?, ?, ?)",
        (recommendation_id, decision, decided_by, decided_at, note),
    )
    conn.commit()
    return get_recommendation(conn, recommendation_id)


def _row_to_dict(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "plant_id": row["plant_id"],
        "recommendation_type": row["recommendation_type"],
        "trigger": row["trigger_name"],
        "evidence": json.loads(row["evidence_json"]),
        "suggested_action": row["suggested_action"],
        "block_timestamp": row["block_timestamp"],
        "created_at": row["created_at"],
    }


def get_recommendation(conn: sqlite3.Connection, recommendation_id: int) -> dict | None:
    row = conn.execute("SELECT * FROM recommendations WHERE id = ?", (recommendation_id,)).fetchone()
    if row is None:
        return None
    result = _row_to_dict(row)

    decision_row = conn.execute(
        "SELECT * FROM recommendation_decisions WHERE recommendation_id = ? ORDER BY id DESC LIMIT 1",
        (recommendation_id,),
    ).fetchone()
    if decision_row is None:
        result.update(status="pending", decided_by=None, decided_at=None, decision_note=None)
    else:
        result.update(
            status=decision_row["decision"],
            decided_by=decision_row["decided_by"],
            decided_at=decision_row["decided_at"],
            decision_note=decision_row["note"],
        )
    return result


def list_recommendations(
    conn: sqlite3.Connection, plant_id: str = None, status: str = None
) -> list:
    """All logged recommendations (newest first), each with its current
    status. Filter by plant_id and/or status (post-hoc, since status is
    derived from the decisions table, not a plain column)."""
    query = "SELECT id FROM recommendations"
    params = []
    if plant_id is not None:
        query += " WHERE plant_id = ?"
        params.append(plant_id)
    query += " ORDER BY id DESC"
    ids = [row["id"] for row in conn.execute(query, params).fetchall()]
    results = [get_recommendation(conn, i) for i in ids]
    if status is not None:
        results = [r for r in results if r["status"] == status]
    return results
