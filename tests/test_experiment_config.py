from dataclasses import FrozenInstanceError, fields
import json
from pathlib import Path

import pytest

from mujoco_sysid_lab.experiment_config import (
    StudyConfig, config_to_dict, load_ground_truth, load_study_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "evidence_v1.json"


def _changed(tmp_path, field_path, value):
    raw = json.loads(CONFIG.read_text())
    cursor = raw
    for key in field_path[:-1]:
        cursor = cursor[key]
    cursor[field_path[-1]] = value
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(raw))
    return path


def test_frozen_full_preregistration_and_truth_isolation():
    study = load_study_config(CONFIG)
    truth = load_ground_truth(study.truth_config_path, study)
    assert study.schema_version == "mujoco-sysid-lab.evidence.v1"
    assert study.noise_seeds == (101, 102, 103, 104, 105)
    assert study.primary_seed == 101
    assert (study.nominal.mass_kg, study.nominal.damping_nm_s_per_rad) == (0.8, 0.04)
    assert (truth.model.mass_kg, truth.model.damping_nm_s_per_rad) == (1.2, 0.15)
    assert truth.delay_study_s == 0.024
    assert dict(truth.mismatch) == {
        "kind": "smooth_coulomb_torque", "torque_nm": 0.08,
        "velocity_scale_rad_s": 0.05, "fitting_model_has_effect": False,
    }
    assert all("truth" not in field.name or field.name == "truth_config_path" for field in fields(StudyConfig))
    assert study.evaluation_window_for("weak_step") == (0.06, 15.94)
    assert study.optimizer["max_iters"] == 200
    assert study.optimizer["diff_step"] == 1e-5
    assert [study.starts[name].delay_s for name in ("S0", "S1", "S2")] == [0.002, 0.05, 0.01]
    assert sum(row["fits"] for row in study.evidence_matrix) == 82
    assert study.criteria["minimum_recovery_seed_count"] == 4
    assert study.criteria["heldout_median_improvement_min"] == 0.2
    assert study.criteria["delay_absolute_error_max_s"] == study.timestep_s
    with pytest.raises(FrozenInstanceError):
        study.primary_seed = 102
    with pytest.raises(TypeError):
        study.sensor_sigmas["qpos"] = 1.0
    with pytest.raises(TypeError):
        study.trajectories["train_prbs"].settings["active_rms_nm"] = 1.0


def test_study_loader_does_not_open_truth(tmp_path):
    path = tmp_path / "evidence_v1.json"
    path.write_text(CONFIG.read_text())
    study = load_study_config(path)
    assert not study.truth_config_path.exists()
    assert study.nominal.mass_kg == 0.8
    snapshot = config_to_dict(study)
    assert snapshot["noise_seeds"] == [101, 102, 103, 104, 105]
    assert "ground_truth" not in snapshot
    assert json.dumps(snapshot, allow_nan=False)


@pytest.mark.parametrize("path,value,error", [
    (["extra_field"], True, "unknown=.*extra_field"),
    (["nominal", "mass_kg"], 2.1, "nominal physical parameters"),
    (["nominal", "timestep_s"], True, "timestep_s"),
    (["nominal", "initial_qpos_rad"], 0.1, "equilibrium prehistory"),
    (["mass_bounds_kg", "lower"], 2.0, "lower < upper"),
    (["mass_bounds_kg", "lower"], 0.0, "physical bounds"),
    (["damping_bounds_nm_s_per_rad", "lower"], -0.01, "physical bounds"),
    (["delay_bounds_s", "lower"], -0.01, "physical bounds"),
    (["starts", "S1", "mass_kg"], 2.1, "starts.S1"),
    (["starts", "S1", "delay_s"], 0.06, "strictly inside delay"),
    (["starts", "S0", "mass_kg"], 0.9, "must equal nominal"),
    (["duration_s"], 8.001, "integer multiple"),
    (["zero_control_prefix_s"], 0.04, "timing requires"),
    (["prehistory_s"], 0.04, "timing requires"),
    (["evaluation_window_s"], [0, 8], "fixed safe"),
    (["noise_seeds"], [101, 101, 103, 104, 105], "unique"),
    (["noise_seeds"], [101, 102, 103, 104, True], "noise_seed"),
    (["primary_seed"], 999, "belong to noise_seeds"),
    (["sensor_sigmas", "qpos"], 0.0, "must be positive"),
    (["noise", "sensor_ids", "qpos"], 1, "noise scheme"),
    (["truth_config_file"], "../hidden.json", "adjacent JSON"),
    (["trajectories", "heldout_chirp", "split"], "train", "partition"),
    (["trajectories", "train_prbs", "trajectory_id"], 0, "partition"),
    (["trajectories", "weak_step", "duration_s"], 8.0, "total rich training duration"),
    (["trajectories", "train_multisine", "settings", "phases_rad"], [0.0], "matching nonzero lengths"),
    (["trajectories", "train_multisine", "settings", "frequencies_hz"], [250, .65, 1.15, 1.8], "Nyquist"),
    (["trajectories", "train_prbs", "settings", "block_duration_s"], 0.251, "integer multiple"),
    (["trajectories", "train_prbs", "settings", "seed_sequence"], [314159, 100, True], "seed_sequence"),
    (["trajectories", "heldout_pulse", "settings", "pulses"], [{"start_s": 0.5, "end_s": 9.0, "amplitude_nm": 1}], "within active support"),
    (["optimizer", "backend"], "custom", "official"),
    (["optimizer", "amplitude_normalization"], True, "conventions"),
    (["optimizer", "diff_step"], 0, "positive"),
    (["optimizer", "rollout_workers"], 2, "one rollout worker"),
    (["noise", "sensor_ids", "qpos"], False, "must be an integer"),
    (["noise", "process_noise"], 0, "must be a boolean"),
    (["phase_lag", "window_margin_s"], .06, "fixed safe window"),
    (["metrics", "bound_hit_fraction"], -0.1, "positive"),
    (["metrics", "split_aggregation"], "pooled_timestep_rmse", "registered metric formulas"),
    (["criteria", "delay_absolute_error_max_s"], .004, "one timestep"),
    (["criteria", "fixed_seed_denominator"], 4, "denominator"),
    (["criteria", "missing_required_seed_counts_as_failure"], 1, "must be a boolean"),
    (["evidence_matrix", 2, "training_ids"], ["heldout_chirp"], "only optimize training"),
    (["evidence_matrix", 2, "fits"], 14, "seeds times starts"),
    (["evidence_matrix", 2, "scenario"], "mismatch", "registered experiment"),
    (["evidence_matrix", 5, "prerequisite"], None, "registered experiment"),
    (["registration", "contacts"], True, "no contacts"),
    (["registration", "dry_friction"], .1, "must be zero"),
])
def test_invalid_registration_is_rejected(tmp_path, path, value, error):
    with pytest.raises(ValueError, match=error):
        load_study_config(_changed(tmp_path, path, value))


@pytest.mark.parametrize("contents,error", [
    ('{"primary_seed": 101, "primary_seed": 102}', "duplicate JSON field"),
    ('{"primary_seed": NaN}', "non-finite JSON constant"),
    ('{"primary_seed": 1e309}', "non-finite number"),
    ('[]', "config must be an object"),
])
def test_strict_json(tmp_path, contents, error):
    path = tmp_path / "invalid.json"
    path.write_text(contents)
    with pytest.raises(ValueError, match=error):
        load_study_config(path)


@pytest.mark.parametrize("path,value,error", [
    (["model", "link_length_m"], 0.6, "share fixed link_length_m"),
    (["model", "mass_kg"], 2.1, "within registered bounds"),
    (["matched_delay_s"], 0.002, "must have zero sensor delay"),
    (["delay_study_s"], 0.061, "outside delay bounds"),
    (["delay_study_s"], 0.023, "integer multiple"),
    (["mismatch", "fitting_model_has_effect"], True, "generator-only"),
    (["mismatch", "torque_nm"], -0.08, "positive"),
])
def test_truth_loader_requires_generator_contract(tmp_path, path, value, error):
    study = load_study_config(CONFIG)
    raw = json.loads(study.truth_config_path.read_text())
    cursor = raw
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    changed = tmp_path / "truth.json"
    changed.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match=error):
        load_ground_truth(changed, study)
