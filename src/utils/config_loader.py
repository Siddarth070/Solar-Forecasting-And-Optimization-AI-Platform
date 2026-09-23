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

import yaml
from pathlib import Path
from functools import lru_cache
from loguru import logger


# Resolve project root relative to this file's location
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH  = PROJECT_ROOT / "configs" / "config.yaml"
PLANTS_DIR   = PROJECT_ROOT / "configs" / "plants"


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
        If configs/plants/<plant_id>.yaml does not exist. Lists the plant
        IDs that do exist, to make the mistake obvious.
    """
    path = PLANTS_DIR / f"{plant_id}.yaml"
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
