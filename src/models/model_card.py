"""
model_card.py — Shared read/write for src/models/model_card.json.
----------------------------------------------------------------------------
Both src/models/train.py and benchmark.py write into the SAME model card:
train.py records what produced the artifact (git SHA, data window, feature
list, its own quick sanity-check metrics); benchmark.py records how well it
actually performs (rolling-origin backtest, final holdout, baselines).
update_model_card() merges into whatever's already there so neither script
has to know about the other's fields.
"""

import json
import subprocess
from pathlib import Path


def _json_default(obj):
    """Numpy scalars (e.g. from a pandas .mean()) aren't JSON-serializable
    by the stdlib json module in all numpy versions -- coerce to a plain
    Python float where possible instead of silently stringifying a number."""
    try:
        return float(obj)
    except (TypeError, ValueError):
        return str(obj)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_CARD_PATH = PROJECT_ROOT / "src" / "models" / "model_card.json"


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def update_model_card(**fields) -> dict:
    """Merge `fields` into the existing model card (creating it if absent)
    and refresh git_sha to the current HEAD every time. Returns the full,
    updated card."""
    card = {}
    if MODEL_CARD_PATH.exists():
        card = json.loads(MODEL_CARD_PATH.read_text())
    card.update(fields)
    card["git_sha"] = _git_sha()
    MODEL_CARD_PATH.write_text(json.dumps(card, indent=2, default=_json_default) + "\n")
    return card
