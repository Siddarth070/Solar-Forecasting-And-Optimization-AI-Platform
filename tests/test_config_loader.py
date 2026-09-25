"""
test_config_loader.py
------------------------
Regression tests for a real path-traversal bug found while building
roadmap P2.3 (plant onboarding): get_plant_config(plant_id) built its
file path as `PLANTS_DIR / f"{plant_id}.yaml"` with zero validation on
plant_id. pathlib's `/` operator silently discards the left operand when
the right operand is itself an absolute path, so
plant_id="/etc/passwd" resolved to Path("/etc/passwd.yaml"), not
anything under PLANTS_DIR -- and every existing plant_id-accepting
endpoint was exposed to it with no validation upstream.

RUN WITH:
  pytest tests/test_config_loader.py -v
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import get_plant_config, list_plant_ids


class TestPathTraversalFix:
    def test_get_plant_config_rejects_absolute_path_plant_id(self):
        with pytest.raises(FileNotFoundError):
            get_plant_config("/etc/passwd")

    def test_get_plant_config_rejects_path_traversal_plant_id(self):
        with pytest.raises(FileNotFoundError):
            get_plant_config("../../../etc/passwd")

    def test_get_plant_config_rejects_malformed_characters(self):
        with pytest.raises(FileNotFoundError):
            get_plant_config("Jaipur.100MW")

    def test_get_plant_config_rejects_empty_string(self):
        with pytest.raises(FileNotFoundError):
            get_plant_config("")


class TestExistingPlantsUnaffected:
    def test_get_plant_config_existing_plants_still_load(self):
        jaipur = get_plant_config("jaipur_100mw")
        assert jaipur["plant_id"] == "jaipur_100mw"
        assert jaipur["capacity"]["ac_capacity_mw"] == 100

        pune = get_plant_config("pune_50mw")
        assert pune["plant_id"] == "pune_50mw"
        assert pune["capacity"]["ac_capacity_mw"] == 50

    def test_list_plant_ids_unaffected(self):
        ids = list_plant_ids()
        assert "jaipur_100mw" in ids
        assert "pune_50mw" in ids
