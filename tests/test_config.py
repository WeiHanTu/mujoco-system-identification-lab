from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path

import numpy as np
import pytest

from mujoco_sysid_lab.config import ParameterBounds, load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "milestone1.json"


def test_registered_config_and_inertia_contract():
    config = load_config(CONFIG_PATH)
    assert (config.ground_truth.mass_kg, config.nominal.mass_kg) == (1.2, 0.8)
    assert (config.ground_truth.damping_nm_s_per_rad, config.nominal.damping_nm_s_per_rad) == (0.15, 0.04)
    assert config.seed == 101
    np.testing.assert_allclose(config.ground_truth.inertia_at_com_kg_m2, [0.0250675, 0.0250675, 0.000135])
    doubled = replace(config.ground_truth, mass_kg=2 * config.ground_truth.mass_kg)
    np.testing.assert_array_equal(
        doubled.inertia_at_com_kg_m2, np.array(config.ground_truth.inertia_at_com_kg_m2) * 2,
    )
    with pytest.raises(FrozenInstanceError):
        config.seed = 102


@pytest.mark.parametrize("field,value", [
    ("mass_kg", 0), ("mass_kg", -1), ("damping_nm_s_per_rad", -0.01),
    ("timestep_s", 0), ("link_length_m", 0), ("link_radius_m", -0.1),
    ("motor_gain_nm_per_unit", 0), ("control_limit_nm", 0),
    ("mass_kg", float("nan")), ("initial_qvel_rad_s", float("inf")),
    ("timestep_s", True),
])
def test_invalid_model_values_are_rejected(field, value):
    config = load_config(CONFIG_PATH)
    with pytest.raises(ValueError, match=field):
        replace(config.ground_truth, **{field: value})


@pytest.mark.parametrize("lower,upper", [(1, 1), (2, 1), (float("nan"), 2), (0, float("inf"))])
def test_invalid_parameter_ranges_are_rejected(lower, upper):
    with pytest.raises(ValueError, match="bounds"):
        ParameterBounds(lower, upper)


def test_bounds_and_fixed_mechanics_are_validated():
    config = load_config(CONFIG_PATH)
    with pytest.raises(ValueError, match="positive"):
        replace(config, mass_bounds_kg=ParameterBounds(0, 2))
    with pytest.raises(ValueError, match="nonnegative"):
        replace(config, damping_bounds_nm_s_per_rad=ParameterBounds(-0.1, 0.5))
    with pytest.raises(ValueError, match="nominal.mass_kg is outside"):
        replace(config, nominal=replace(config.nominal, mass_kg=2.1))
    with pytest.raises(ValueError, match="nominal.damping_nm_s_per_rad is outside"):
        replace(config, nominal=replace(config.nominal, damping_nm_s_per_rad=0.6))
    with pytest.raises(ValueError, match="must share fixed timestep_s"):
        replace(config, nominal=replace(config.nominal, timestep_s=0.004))


@pytest.mark.parametrize("duration", [0, -1, 0.001, 1.0005, float("inf")])
def test_duration_must_end_on_simulation_grid(duration):
    with pytest.raises(ValueError, match="duration_s"):
        replace(load_config(CONFIG_PATH), duration_s=duration)


@pytest.mark.parametrize("seed", [-1, 1.5, True])
def test_seed_must_be_nonnegative_integer(seed):
    with pytest.raises(ValueError, match="seed"):
        replace(load_config(CONFIG_PATH), seed=seed)


def test_json_fields_and_schema_are_strict(tmp_path):
    raw = json.loads(CONFIG_PATH.read_text())
    raw["unregistered_knob"] = 0.1
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="unknown=.*unregistered_knob"):
        load_config(path)
    del raw["unregistered_knob"]
    raw["schema_version"] = "unknown-v2"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="unsupported config schema"):
        load_config(path)
    raw["schema_version"] = "mujoco-sysid-lab.milestone1.v1"
    del raw["ground_truth"]["mass_kg"]
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="missing=.*mass_kg"):
        load_config(path)


@pytest.mark.parametrize("contents,error", [
    ('{"seed": 101, "seed": 102}', "duplicate JSON field"),
    ('{"seed": NaN}', "non-finite JSON constant"),
    ('[]', "config must be an object"),
])
def test_noncanonical_json_is_rejected(tmp_path, contents, error):
    path = tmp_path / "invalid.json"
    path.write_text(contents)
    with pytest.raises(ValueError, match=error):
        load_config(path)
