"""
config_loader.py
----------------
Reads configs/config.yaml and exposes settings as a typed object.

WHY THIS EXISTS:
  Every module in the project imports get_config() instead of
  reading the YAML directly. This means:
  1. Config is loaded once and cached
  2. You change one file (config.yaml) to change all behaviour
  3. Tests can swap configs without touching source code

USAGE:
  from src.utils.config_loader import get_config
  cfg = get_config()
  lat = cfg['location']['latitude']
"""

import yaml
from pathlib import Path
from functools import lru_cache
from loguru import logger


# Resolve project root relative to this file's location
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH  = PROJECT_ROOT / "configs" / "config.yaml"


@lru_cache(maxsize=1)
def get_config() -> dict:
    """
    Load and cache the master config.

    Returns
    -------
    dict
        Full config dictionary from configs/config.yaml

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
    logger.info(f"Target location: {config['location']['name']}, {config['location']['state']}")

    return config


def get_location(config: dict | None = None) -> dict:
    """Convenience function — returns just the location block."""
    cfg = config or get_config()
    return cfg["location"]


def get_data_sources(config: dict | None = None) -> dict:
    """Convenience function — returns just the data_sources block."""
    cfg = config or get_config()
    return cfg["data_sources"]
