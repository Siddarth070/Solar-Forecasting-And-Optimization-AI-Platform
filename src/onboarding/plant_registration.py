"""
plant_registration.py — Self-serve plant onboarding (roadmap P2.3).

WHY THIS EXISTS:
  Per the roadmap: "self-serve capture of everything in P1.1 plus COD,
  module type, inverter count, grid export limit," with acceptance
  criterion "a new plant is live in under 30 minutes without your
  involvement." Before this module, the ONLY way to add a plant was for
  a developer to hand-author and commit a configs/plants/<id>.yaml file
  -- there was no validation, no registration API, nothing. This module
  writes that same file, validated, so a freshly onboarded plant is
  immediately usable by every existing endpoint (they all read via
  src.utils.config_loader.get_plant_config, the exact function this
  writes for) with no restart and no code change.

SCOPED TO CONFIGURATION, NOT DATA: this does NOT depend on roadmap P2.1
  (CSV/Excel historical data upload, not yet built -- blocked on real
  customer files). It only captures the plant metadata every endpoint's
  plant_config argument needs; a freshly onboarded plant has no
  historical generation data of its own until P2.1 exists or a caller
  supplies readings directly (e.g. to POST /quality/check).

CREATE-ONLY: there is no update/edit function here. get_plant_config()
  is @lru_cache'd; safely invalidating that cache for an in-place edit
  is separate future work (see README Known Gaps) -- the roadmap itself
  only asks for "a NEW plant," not editing an existing one.

SECURITY: plant_id is validated via src.utils.config_loader's
  PLANT_ID_PATTERN (a strict allowlist, reused here rather than a second
  copy) PLUS an independent resolved-path containment check before any
  file is written -- the same two-layer defense config_loader.py's own
  get_plant_config() now applies on the read side, for the identical
  reason: this module is a file-WRITE path, and a client-supplied
  plant_id must never be trusted to build a filesystem path directly.
"""

import datetime as _dt
import os
import tempfile
import zoneinfo
from pathlib import Path

import yaml
from loguru import logger

from src.regulatory import dsm, grid_code
from src.utils import config_loader
from src.utils.config_loader import PLANT_ID_PATTERN

DSM_RULESET_ID = "CERC_DSM_2024_Reg8(4)"
GRID_CODE_RULESET_ID = "CERC_IEGC_2023_Reg49"
# Fixed -- only one ruleset of each is implemented today
# (src/regulatory/dsm.py, src/regulatory/grid_code.py). Auto-filled,
# never caller input; a real plant operator can't be expected to know
# these internal identifier strings. Update these constants, not the
# onboarding schema, if a second ruleset is ever added.


class PlantAlreadyRegisteredError(Exception):
    """register_plant() found configs/plants/<plant_id>.yaml already
    exists. This module is create-only by design -- editing an existing
    plant is explicitly out of scope (see module docstring)."""


def validate_plant_id(plant_id: str) -> None:
    """Raises ValueError for a malformed plant_id. Deliberately ValueError,
    not FileNotFoundError: this is a CREATE flow, where a malformed id is
    a 422 validation error, distinct from config_loader's READ flow
    (get_plant_config), which treats the same malformed input as a 404
    (an already-existing convention this module doesn't touch)."""
    if not PLANT_ID_PATTERN.fullmatch(plant_id):
        raise ValueError(
            f"Invalid plant_id {plant_id!r} -- must match "
            f"{PLANT_ID_PATTERN.pattern} (lowercase letters, digits, "
            f"underscores only, 2-50 chars, starting with a letter)."
        )


def _resolve_target_path(plants_dir: Path, plant_id: str) -> Path:
    """Same resolved-path containment check as
    config_loader._resolve_plant_path, raising ValueError to match this
    module's create-flow semantics (a small, deliberate duplication of a
    3-line check rather than sharing one helper with two different
    exception types)."""
    plants_dir_resolved = Path(plants_dir).resolve()
    target = (plants_dir_resolved / f"{plant_id}.yaml").resolve()
    if target.parent != plants_dir_resolved:
        raise ValueError(f"plant_id {plant_id!r} does not resolve inside {plants_dir_resolved}.")
    return target


def _validate_location(location: dict) -> None:
    if not location.get("name"):
        raise ValueError("location.name is required and must be non-empty.")
    if not location.get("state"):
        raise ValueError("location.state is required and must be non-empty.")
    lat = location.get("latitude")
    if lat is None or not (-90 <= lat <= 90):
        raise ValueError(f"location.latitude must be in [-90, 90], got {lat!r}.")
    lon = location.get("longitude")
    if lon is None or not (-180 <= lon <= 180):
        raise ValueError(f"location.longitude must be in [-180, 180], got {lon!r}.")
    elevation = location.get("elevation_m")
    if elevation is not None and not (-500 <= elevation <= 9000):
        raise ValueError(f"location.elevation_m must be in [-500, 9000], got {elevation!r}.")
    tz = location.get("timezone")
    if not tz:
        raise ValueError("location.timezone is required and must be non-empty.")
    try:
        zoneinfo.ZoneInfo(tz)
    except Exception as e:
        raise ValueError(f"location.timezone {tz!r} is not a valid IANA timezone: {e}")


def _validate_capacity(capacity: dict) -> None:
    for field in ("ac_capacity_mw", "dc_capacity_mw", "panel_area_m2"):
        value = capacity.get(field)
        if value is None or value <= 0:
            raise ValueError(f"capacity.{field} must be > 0, got {value!r}.")
    for field in ("panel_efficiency", "performance_ratio"):
        # Required, never defaulted -- src/attribution/loss.py and
        # src/evaluation/baselines.py index these with no fallback, so
        # onboarding must not silently inject a generic value.
        value = capacity.get(field)
        if value is None or not (0 < value <= 1):
            raise ValueError(f"capacity.{field} must be in (0, 1], got {value!r}.")
    temp_coef = capacity.get("temperature_coefficient")
    if temp_coef is None or not (-0.02 <= temp_coef <= 0):
        raise ValueError(
            f"capacity.temperature_coefficient must be in [-0.02, 0] "
            f"(a sanity bound, not a regulatory fact), got {temp_coef!r}."
        )


def _validate_grid(grid: dict, capacity: dict) -> None:
    export_limit = grid.get("export_limit_mw")
    if export_limit is None or export_limit <= 0:
        raise ValueError(f"grid.export_limit_mw must be > 0, got {export_limit!r}.")
    if export_limit > capacity["ac_capacity_mw"]:
        # Unusual but not invalid (e.g. quoted for a planned capacity
        # expansion) -- this module captures facts, it doesn't
        # adjudicate plant engineering.
        logger.warning(
            f"grid.export_limit_mw ({export_limit}) exceeds "
            f"capacity.ac_capacity_mw ({capacity['ac_capacity_mw']}) -- unusual, "
            f"but not rejected."
        )


def _validate_equipment(equipment: dict) -> None:
    cod = equipment.get("commercial_operation_date")
    if not cod:
        raise ValueError("equipment.commercial_operation_date is required.")
    try:
        _dt.date.fromisoformat(cod)
    except ValueError as e:
        raise ValueError(
            f"equipment.commercial_operation_date {cod!r} must be ISO 8601 "
            f"(YYYY-MM-DD): {e}"
        )
    module_type = equipment.get("module_type")
    if not module_type or len(module_type) > 200:
        raise ValueError("equipment.module_type must be non-empty and <= 200 chars.")
    inverter_count = equipment.get("inverter_count")
    if inverter_count is None or inverter_count < 1:
        raise ValueError(f"equipment.inverter_count must be >= 1, got {inverter_count!r}.")


def _validate_regulatory(regulatory: dict) -> None:
    seller_category = regulatory.get("seller_category")
    if seller_category not in dsm.SELLER_CATEGORIES:
        raise ValueError(
            f"regulatory.seller_category must be one of "
            f"{sorted(dsm.SELLER_CATEGORIES)}, got {seller_category!r}."
        )
    contract_rate = regulatory.get("contract_rate_rs_per_kwh")
    if contract_rate is not None and contract_rate <= 0:
        raise ValueError(f"regulatory.contract_rate_rs_per_kwh must be > 0, got {contract_rate!r}.")
    transaction_type = regulatory.get("transaction_type")
    if transaction_type is not None and transaction_type not in grid_code.KNOWN_TRANSACTION_TYPES:
        raise ValueError(
            f"regulatory.transaction_type must be one of "
            f"{sorted(grid_code.KNOWN_TRANSACTION_TYPES)}, got {transaction_type!r}."
        )


def build_plant_config(
    plant_id: str,
    name: str,
    location: dict,
    capacity: dict,
    grid: dict,
    equipment: dict,
    regulatory: dict,
) -> dict:
    """
    Pure, disk-free: validates every field and assembles the full dict
    get_plant_config() would return. Two field-omission rules matter for
    correctness here (found by reading every downstream consumer, not
    assumed):

    - location["elevation_m"]: omitted entirely if not supplied. Every
      consumer does `.get("elevation_m", 0.0)`, which only applies that
      default when the key is ABSENT -- an explicit `null` would pass
      None straight into pvlib and crash.
    - regulatory["revision_windows"]["transaction_type"]: omitted
      entirely if not supplied. src/api/main.py resolves it via
      `.get("transaction_type", "bilateral")` -- same absent-vs-null
      distinction; an explicit `null` would silently disable the
      documented "defaults to bilateral" behavior.

    contract_rate_rs_per_kwh, by contrast, is safe to write as an
    explicit None when absent, since every consumer checks `is None`
    directly, not `.get(key, default)`.
    """
    validate_plant_id(plant_id)
    if not name:
        raise ValueError("name is required and must be non-empty.")
    _validate_location(location)
    _validate_capacity(capacity)
    _validate_grid(grid, capacity)
    _validate_equipment(equipment)
    _validate_regulatory(regulatory)

    location_out = {
        "name": location["name"],
        "state": location["state"],
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        "timezone": location["timezone"],
    }
    if location.get("elevation_m") is not None:
        location_out["elevation_m"] = location["elevation_m"]

    capacity_out = {
        "ac_capacity_mw": capacity["ac_capacity_mw"],
        "dc_capacity_mw": capacity["dc_capacity_mw"],
        "panel_efficiency": capacity["panel_efficiency"],
        "temperature_coefficient": capacity["temperature_coefficient"],
        "performance_ratio": capacity["performance_ratio"],
        "panel_area_m2": capacity["panel_area_m2"],
        "tilt_deg": None,   # not exposed as an onboarding input -- unused today
        "azimuth_deg": None,  # not exposed as an onboarding input -- unused today
    }

    grid_out = {
        "sldc": None,
        "rldc": None,
        "ists_or_instate": None,
        "qca_role": None,
        "metering_point": None,
        "export_limit_mw": grid["export_limit_mw"],
    }

    equipment_out = {
        "commercial_operation_date": equipment["commercial_operation_date"],
        "module_type": equipment["module_type"],
        "inverter_count": equipment["inverter_count"],
    }

    revision_windows_out = {"grid_code_ruleset_id": GRID_CODE_RULESET_ID}
    if regulatory.get("transaction_type") is not None:
        revision_windows_out["transaction_type"] = regulatory["transaction_type"]

    regulatory_out = {
        "dsm_ruleset_id": DSM_RULESET_ID,
        "seller_category": regulatory["seller_category"],
        "contract_rate_rs_per_kwh": regulatory.get("contract_rate_rs_per_kwh"),
        "schedule_format": None,  # not exposed as an onboarding input -- unused today
        "revision_windows": revision_windows_out,
    }

    return {
        "plant_id": plant_id,
        "name": name,
        "location": location_out,
        "capacity": capacity_out,
        "grid": grid_out,
        "equipment": equipment_out,
        "regulatory": regulatory_out,
    }


def _write_yaml_atomic(path: Path, data: dict) -> None:
    """Atomic write: temp file in the SAME directory (so os.replace is
    atomic -- same filesystem), fsync'd before rename, cleaned up on any
    failure. Note: yaml.safe_dump doesn't preserve/produce the rich
    inline comments the two hand-authored demo YAMLs have -- an accepted,
    cosmetic difference, not a data-loss concern."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def register_plant(
    plant_id: str,
    name: str,
    location: dict,
    capacity: dict,
    grid: dict,
    equipment: dict,
    regulatory: dict,
    plants_dir: Path | None = None,
) -> dict:
    """
    Validate and write a NEW configs/plants/<plant_id>.yaml. Raises
    PlantAlreadyRegisteredError if plant_id is already registered (never
    silently overwrites), or ValueError for any invalid field.

    Ordering is deliberate: format check (cheapest, must always run) ->
    duplicate check (cheap, and "don't silently overwrite" is the single
    most safety-critical property of this function) -> full semantic
    validation (only worth doing once we know the result won't be
    thrown away).
    """
    validate_plant_id(plant_id)
    # config_loader.PLANTS_DIR read as a module attribute at CALL TIME
    # (not `from ... import PLANTS_DIR` bound once at import time), so a
    # test monkeypatching config_loader.PLANTS_DIR is honored -- the same
    # reason get_plant_config/list_plant_ids already work correctly
    # under monkeypatching today.
    target_dir = plants_dir if plants_dir is not None else config_loader.PLANTS_DIR
    target_path = _resolve_target_path(target_dir, plant_id)
    if target_path.exists():
        raise PlantAlreadyRegisteredError(f"Plant {plant_id!r} is already registered.")

    config = build_plant_config(plant_id, name, location, capacity, grid, equipment, regulatory)
    _write_yaml_atomic(target_path, config)
    logger.success(f"Plant registered: {plant_id} -> {target_path}")
    return config
