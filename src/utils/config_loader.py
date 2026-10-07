"""
config_loader.py
----------------
Reads configs/config.yaml (global, plant-agnostic settings) and
configs/plants/<plant_id>.yaml (per-plant location, capacity, grid and
regulatory identifiers) and exposes both as dicts.

WHY THIS EXISTS:
  Every module in the project imports get_config()/get_plant_config()
  instead of reading YAML directly. This means:
  1. Config is loaded once per plant and cached
  2. You change one file to change all behaviour
  3. Tests can swap configs without touching source code

WHY PLANT CONFIG IS SEPARATE FROM THE GLOBAL CONFIG (roadmap P1.1):
  configs/config.yaml used to hardcode one Jaipur lat/lon under `location`,
  and one plant's capacity under `solar_plant` — so no module could serve
  more than one plant. Location/capacity/grid/regulatory identifiers now
  live per-plant in configs/plants/<plant_id>.yaml; the global config only
  holds settings that are genuinely the same for every plant (data source
  URLs, weather variables to fetch, logging, validation bounds).

USAGE:
  from src.utils.config_loader import get_config, get_plant_config
  cfg = get_config()
  plant = get_plant_config("jaipur_100mw")
  lat = plant["location"]["latitude"]
"""

import re
import yaml
from pathlib import Path
from functools import lru_cache
from loguru import logger


# Resolve project root relative to this file's location
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH  = PROJECT_ROOT / "configs" / "config.yaml"
PLANTS_DIR   = PROJECT_ROOT / "configs" / "plants"

PLANT_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]{1,49}$")
# Lowercase-letter start, then lowercase letters/digits/underscores,
# 2-50 chars total. Deliberately an ALLOWLIST, not a blocklist -- a
# string matching this cannot contain "/", "..", a drive letter, or a
# null byte at all, so it can't express a path-traversal payload
# regardless of what any single blocklist check might miss.


@lru_cache(maxsize=1)
def get_config() -> dict:
    """
    Load and cache the global, plant-agnostic config.

    Returns
    -------
    dict
        Config dictionary from configs/config.yaml

    Raises
    ------
    FileNotFoundError
        If configs/config.yaml does not exist
    """
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Config file not found at {CONFIG_PATH}. "
            "Ensure you are running from the project root."
        )

    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Config loaded from {CONFIG_PATH}")
    logger.info(f"Project: {config['project']['name']} v{config['project']['version']}")

    return config


def list_plant_ids() -> list[str]:
    """All plant IDs with a configs/plants/<plant_id>.yaml file."""
    if not PLANTS_DIR.exists():
        return []
    return sorted(p.stem for p in PLANTS_DIR.glob("*.yaml"))


def _resolve_plant_path(plant_id: str) -> Path:
    """
    Resolve plant_id to configs/plants/<plant_id>.yaml, defending against
    path traversal via TWO independent checks: a strict allowlist regex
    (PLANT_ID_PATTERN), and a resolved-path containment check.

    This is a REAL bug fix, not a theoretical hardening: pathlib's `/`
    operator silently discards the left operand entirely when the right
    operand is itself an absolute path, so building the path as
    `PLANTS_DIR / f"{plant_id}.yaml"` with plant_id="/etc/passwd" used to
    resolve to Path("/etc/passwd.yaml"), not a path under PLANTS_DIR --
    and every existing plant_id-accepting endpoint in src/api/main.py
    passed its plant_id straight into this function with no validation
    at all. A relative plant_id like "../../../etc/passwd" has the same
    problem via ordinary ".." traversal.

    Raises FileNotFoundError (not ValueError) for BOTH a malformed
    plant_id and a genuinely-missing one -- deliberately unified under
    the one exception type every existing call site already catches
    (`except FileNotFoundError: raise HTTPException(404, ...)`), so this
    fix requires touching zero of those call sites: from every caller's
    perspective, a malformed ID and a nonexistent ID both mean "no such
    plant."
    """
    if not PLANT_ID_PATTERN.fullmatch(plant_id):
        raise FileNotFoundError(
            f"Invalid plant_id {plant_id!r} -- must match "
            f"{PLANT_ID_PATTERN.pattern} (lowercase letters, digits, "
            f"underscores only). Available plant IDs: {list_plant_ids()}"
        )
    plants_dir_resolved = PLANTS_DIR.resolve()
    target = (plants_dir_resolved / f"{plant_id}.yaml").resolve()
    if target.parent != plants_dir_resolved:
        raise FileNotFoundError(f"plant_id {plant_id!r} is not a valid plant ID.")
    return target


@lru_cache(maxsize=None)
def get_plant_config(plant_id: str) -> dict:
    """
    Load and cache one plant's config.

    Parameters
    ----------
    plant_id : str
        Matches a filename under configs/plants/ (without .yaml).

    Returns
    -------
    dict
        Plant config: location, capacity, grid, regulatory blocks.

    Raises
    ------
    FileNotFoundError
        If plant_id is malformed (see _resolve_plant_path) or
        configs/plants/<plant_id>.yaml does not exist. Lists the plant
        IDs that do exist, to make the mistake obvious.
    """
    path = _resolve_plant_path(plant_id)
    if not path.exists():
        available = list_plant_ids()
        raise FileNotFoundError(
            f"No plant config at {path}. Available plant IDs: {available}"
        )

    with open(path, "r") as f:
        plant = yaml.safe_load(f)

    logger.info(
        f"Plant config loaded: {plant['plant_id']} — "
        f"{plant['location']['name']}, {plant['location']['state']} "
        f"({plant['capacity']['ac_capacity_mw']} MW AC)"
    )
    return plant


def get_data_sources(config: dict | None = None) -> dict:
    """Convenience function — returns just the data_sources block."""
    cfg = config or get_config()
    return cfg["data_sources"]
