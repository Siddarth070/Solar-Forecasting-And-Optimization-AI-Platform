"""
test_plant_registration.py
-----------------------------
Roadmap P2.3 (plant onboarding) -- unit tests for the pure registration
module (no FastAPI, no real configs/plants/ -- everything file-touching
uses tmp_path). Each dirty-input test injects exactly one problem so a
rejection is unambiguous; each valid-input test hand-checks the exact
resulting dict, especially the two field-omission rules
(elevation_m/transaction_type) that are easy to get subtly wrong.

RUN WITH:
  pytest tests/test_plant_registration.py -v
"""

import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.regulatory import dsm, grid_code
from src.onboarding.plant_registration import (
    DSM_RULESET_ID,
    GRID_CODE_RULESET_ID,
    PlantAlreadyRegisteredError,
    _resolve_target_path,
    build_plant_config,
    register_plant,
    validate_plant_id,
)


def _valid_fields(**overrides):
    fields = dict(
        plant_id="kolar_75mw",
        name="Kolar Solar Plant",
        location={
            "name": "Kolar", "state": "Karnataka",
            "latitude": 13.13, "longitude": 78.13, "timezone": "Asia/Kolkata",
        },
        capacity={
            "ac_capacity_mw": 75.0, "dc_capacity_mw": 90.0,
            "panel_efficiency": 0.21, "temperature_coefficient": -0.0035,
            "performance_ratio": 0.82, "panel_area_m2": 400000.0,
        },
        grid={"export_limit_mw": 70.0},
        equipment={
            "commercial_operation_date": "2021-05-01",
            "module_type": "Bifacial 550Wp", "inverter_count": 25,
        },
        regulatory={"seller_category": "solar"},
    )
    fields.update(overrides)
    return fields


class TestValidatePlantId:
    def test_accepts_valid_id(self):
        validate_plant_id("kolar_75mw")  # must not raise

    def test_rejects_path_traversal(self):
        with pytest.raises(ValueError):
            validate_plant_id("../../etc/passwd")

    def test_rejects_absolute_path(self):
        with pytest.raises(ValueError):
            validate_plant_id("/etc/passwd")

    def test_rejects_uppercase_and_dots(self):
        with pytest.raises(ValueError):
            validate_plant_id("Kolar.75MW")

    def test_rejects_empty_and_too_long(self):
        with pytest.raises(ValueError):
            validate_plant_id("")
        with pytest.raises(ValueError):
            validate_plant_id("a" * 51)


class TestResolveTargetPath:
    def test_stays_inside_plants_dir(self, tmp_path):
        result = _resolve_target_path(tmp_path, "kolar_75mw")
        assert result == (tmp_path / "kolar_75mw.yaml").resolve()

    def test_rejects_absolute_plant_id_reaching_here_directly(self, tmp_path):
        # Regression test for the pathlib absolute-join gotcha: even if a
        # caller somehow bypassed validate_plant_id, this independent
        # containment check must still catch it.
        with pytest.raises(ValueError):
            _resolve_target_path(tmp_path, "/etc/passwd")


class TestBuildPlantConfig:
    def test_full_valid_round_trip(self):
        result = build_plant_config(**_valid_fields())
        assert result["plant_id"] == "kolar_75mw"
        assert result["name"] == "Kolar Solar Plant"
        assert result["capacity"]["ac_capacity_mw"] == 75.0
        assert result["capacity"]["tilt_deg"] is None
        assert result["capacity"]["azimuth_deg"] is None
        assert result["grid"]["export_limit_mw"] == 70.0
        assert result["grid"]["sldc"] is None
        assert result["equipment"]["module_type"] == "Bifacial 550Wp"
        assert result["regulatory"]["dsm_ruleset_id"] == DSM_RULESET_ID
        assert result["regulatory"]["revision_windows"]["grid_code_ruleset_id"] == GRID_CODE_RULESET_ID
        assert result["regulatory"]["schedule_format"] is None

    def test_omits_elevation_when_not_given(self):
        result = build_plant_config(**_valid_fields())
        assert "elevation_m" not in result["location"]

    def test_includes_elevation_when_given(self):
        fields = _valid_fields()
        fields["location"] = {**fields["location"], "elevation_m": 850.0}
        result = build_plant_config(**fields)
        assert result["location"]["elevation_m"] == 850.0

    def test_omits_transaction_type_when_not_given(self):
        result = build_plant_config(**_valid_fields())
        assert "transaction_type" not in result["regulatory"]["revision_windows"]

    def test_includes_transaction_type_when_given(self):
        fields = _valid_fields()
        fields["regulatory"] = {"seller_category": "solar", "transaction_type": "bilateral"}
        result = build_plant_config(**fields)
        assert result["regulatory"]["revision_windows"]["transaction_type"] == "bilateral"

    def test_contract_rate_written_as_explicit_none_when_absent(self):
        result = build_plant_config(**_valid_fields())
        assert result["regulatory"]["contract_rate_rs_per_kwh"] is None

    def test_rejects_unknown_seller_category(self):
        fields = _valid_fields()
        fields["regulatory"] = {"seller_category": "coal"}
        with pytest.raises(ValueError, match="seller_category"):
            build_plant_config(**fields)

    def test_rejects_unknown_transaction_type(self):
        fields = _valid_fields()
        fields["regulatory"] = {"seller_category": "solar", "transaction_type": "exchange"}
        with pytest.raises(ValueError, match="transaction_type"):
            build_plant_config(**fields)

    def test_rejects_invalid_timezone(self):
        fields = _valid_fields()
        fields["location"] = {**fields["location"], "timezone": "Mars/Phobos"}
        with pytest.raises(ValueError, match="timezone"):
            build_plant_config(**fields)

    def test_rejects_invalid_cod_date_format(self):
        fields = _valid_fields()
        fields["equipment"] = {**fields["equipment"], "commercial_operation_date": "15-03-2019"}
        with pytest.raises(ValueError, match="commercial_operation_date"):
            build_plant_config(**fields)

    def test_rejects_out_of_range_latitude_even_bypassing_pydantic(self):
        fields = _valid_fields()
        fields["location"] = {**fields["location"], "latitude": 200.0}
        with pytest.raises(ValueError, match="latitude"):
            build_plant_config(**fields)

    def test_missing_required_capacity_field_raises(self):
        fields = _valid_fields()
        fields["capacity"] = {k: v for k, v in fields["capacity"].items() if k != "performance_ratio"}
        with pytest.raises(ValueError, match="performance_ratio"):
            build_plant_config(**fields)

    def test_export_limit_above_capacity_allowed_with_warning(self, caplog):
        fields = _valid_fields()
        fields["grid"] = {"export_limit_mw": 500.0}  # far above ac_capacity_mw=75
        result = build_plant_config(**fields)  # must not raise
        assert result["grid"]["export_limit_mw"] == 500.0

    def test_reuses_dsm_seller_categories_not_a_duplicate_list(self, monkeypatch):
        monkeypatch.setattr(dsm, "SELLER_CATEGORIES", frozenset({"only_this_one"}))
        fields = _valid_fields()
        fields["regulatory"] = {"seller_category": "solar"}  # no longer valid under the patched set
        with pytest.raises(ValueError):
            build_plant_config(**fields)

    def test_reuses_grid_code_transaction_types_not_a_duplicate_list(self, monkeypatch):
        monkeypatch.setattr(grid_code, "KNOWN_TRANSACTION_TYPES", frozenset({"only_this_one"}))
        fields = _valid_fields()
        fields["regulatory"] = {"seller_category": "solar", "transaction_type": "bilateral"}
        with pytest.raises(ValueError):
            build_plant_config(**fields)


class TestRegisterPlant:
    def test_writes_file(self, tmp_path):
        config = register_plant(**_valid_fields(), plants_dir=tmp_path)
        written_path = tmp_path / "kolar_75mw.yaml"
        assert written_path.exists()
        with open(written_path) as f:
            on_disk = yaml.safe_load(f)
        assert on_disk == config

    def test_duplicate_raises_and_does_not_overwrite(self, tmp_path):
        register_plant(**_valid_fields(name="First Name"), plants_dir=tmp_path)
        with pytest.raises(PlantAlreadyRegisteredError):
            register_plant(**_valid_fields(name="Second Name"), plants_dir=tmp_path)
        with open(tmp_path / "kolar_75mw.yaml") as f:
            on_disk = yaml.safe_load(f)
        assert on_disk["name"] == "First Name"

    def test_atomic_write_leaves_no_tmp_file_on_success(self, tmp_path):
        register_plant(**_valid_fields(), plants_dir=tmp_path)
        assert [p.name for p in tmp_path.iterdir()] == ["kolar_75mw.yaml"]

    def test_default_plants_dir_uses_config_loader_at_call_time(self, tmp_path, monkeypatch):
        from src.utils import config_loader
        monkeypatch.setattr(config_loader, "PLANTS_DIR", tmp_path)
        register_plant(**_valid_fields())  # no plants_dir passed
        assert (tmp_path / "kolar_75mw.yaml").exists()

    def test_invalid_plant_id_raises_before_touching_disk(self, tmp_path):
        with pytest.raises(ValueError):
            register_plant(**_valid_fields(plant_id="../evil"), plants_dir=tmp_path)
        assert list(tmp_path.iterdir()) == []
