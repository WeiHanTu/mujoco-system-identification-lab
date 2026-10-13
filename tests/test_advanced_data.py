from dataclasses import fields, replace
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from mujoco_sysid_lab.advanced_data import (
    ARRAY_NAMES, SCENARIOS, apply_sensor_delay, create_scenario,
    load_scenario, load_scenario_training, simulate_mismatch, validate_scenario, write_scenario,
)
from mujoco_sysid_lab.dataset import ARRAY_NAMES as M2_ARRAY_NAMES, TRAIN_ARRAY_NAMES, TrainingSequence, TrajectoryRecord, generate_dataset
from mujoco_sysid_lab.experiment_config import load_ground_truth, load_study_config
from mujoco_sysid_lab.provenance import collect_provenance, sha256_file
from mujoco_sysid_lab.simulation import compile_model, simulate


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "model" / "pendulum.xml"


@pytest.fixture(scope="module")
def study():
    return load_study_config(ROOT / "configs" / "evidence_v1.json")


@pytest.fixture(scope="module")
def truth(study):
    return load_ground_truth(study.truth_config_path, study)


@pytest.fixture(scope="module")
def scenarios(study, truth):
    return {name: create_scenario(study, truth, MODEL, 101, name) for name in SCENARIOS}


@pytest.fixture(scope="module")
def provenance():
    return collect_provenance(ROOT, ROOT / "configs" / "evidence_v1.json", MODEL,
                              [".venv/bin/python", "-m", "pytest", "tests/test_advanced_data.py"])


def test_matched_records_exactly_preserve_all_milestone2_arrays(study, truth, scenarios):
    original = generate_dataset(study, truth, MODEL, 101)
    for name in original.records:
        for key in M2_ARRAY_NAMES:
            np.testing.assert_array_equal(scenarios["matched"].records[name].arrays[key],
                                          original.records[name].arrays[key])
    assert scenarios["matched"].delay_s == 0


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_all_scenarios_deterministic_shapes_finite_and_immutable(study, truth, scenarios, scenario):
    first = scenarios[scenario]
    second = create_scenario(study, truth, MODEL, 101, scenario)
    assert first.metadata == second.metadata
    assert first.training_ids == (("weak_step",) if scenario == "weak" else ("train_multisine", "train_prbs"))
    assert tuple(first.heldout) == ("heldout_chirp", "heldout_pulse")
    for name, record in first.records.items():
        assert set(record.arrays) == ARRAY_NAMES
        samples = 8001 if name == "weak_step" else 4001
        assert record.arrays["timestamps"].shape == (samples,)
        assert record.arrays["measured_qpos"].shape == record.arrays["measured_qvel"].shape == (samples, 1)
        assert record.arrays["timestamps"][-1] == (16 if name == "weak_step" else 8)
        for key, array in record.arrays.items():
            assert array.dtype == np.float64 and np.isfinite(array).all()
            assert not array.flags.writeable
            np.testing.assert_array_equal(array, second.records[name].arrays[key])
        assert record.metadata == second.records[name].metadata
        with pytest.raises(TypeError):
            record.metadata["noise"]["qpos"]["sigma"] = 999
    assert {field.name for field in fields(TrainingSequence)} == {"trajectory_name", "trajectory_id", *TRAIN_ARRAY_NAMES}
    assert tuple(sequence.trajectory_name for sequence in first.training.sequences) == first.training_ids
    assert not hasattr(first.training, "metadata")


def test_all_rich_scenarios_reuse_identical_registered_innovations(scenarios):
    for name, matched in scenarios["matched"].records.items():
        for sensor in ("qpos", "qvel"):
            for scenario in ("delayed", "mismatch"):
                record = scenarios[scenario].records[name]
                np.testing.assert_array_equal(record.arrays["innovation_" + sensor], matched.arrays["innovation_" + sensor])
                assert record.metadata["noise"][sensor] == matched.metadata["noise"][sensor]
                sigma = record.metadata["noise"][sensor]["sigma"]
                np.testing.assert_array_equal(record.arrays["measured_" + sensor],
                                              record.arrays["observed_clean_" + sensor] + sigma * matched.arrays["innovation_" + sensor])


def test_delayed_data_uses_clean_history_before_paired_noise(scenarios):
    delayed, matched = scenarios["delayed"], scenarios["matched"]
    assert delayed.delay_s == .024
    for name, record in delayed.records.items():
        arrays = record.arrays
        for sensor in ("qpos", "qvel"):
            np.testing.assert_array_equal(arrays["clean_" + sensor], matched.records[name].arrays["clean_" + sensor])
            # Registered lag is twelve complete sensor intervals. Interpolation
            # clock subtraction differs from integer-grid indexing by an ULP.
            # Bound its effect using the signal's largest sampled slope rather
            # than changing the 2 ms scientific delay-recovery threshold.
            expected = np.vstack((np.zeros((12, 1)), arrays["clean_" + sensor][:-12]))
            clock_error = np.max(np.abs(arrays["timestamps"][12:] - .024 - arrays["timestamps"][:-12]))
            slope_bound = np.max(np.abs(np.diff(arrays["clean_" + sensor][:, 0]) / .002))
            roundoff = 8 * np.finfo(np.float64).eps * np.max(np.abs(arrays["clean_" + sensor]))
            np.testing.assert_allclose(arrays["observed_clean_" + sensor], expected,
                                       atol=slope_bound * clock_error + roundoff, rtol=0)
            assert not np.array_equal(arrays["observed_clean_" + sensor], arrays["clean_" + sensor])


def test_delay_sign_with_linear_fixture_and_no_extrapolation():
    history = np.arange(31) * .002 - .06
    history[-1] = 0
    times = np.arange(51) * .002
    delayed = apply_sensor_delay(times, times[:, None], .024, history, history[:, None])
    np.testing.assert_allclose(delayed[:, 0], times - .024, rtol=0, atol=1e-16)
    for invalid in (-.002, .060001, np.nan, True):
        with pytest.raises(ValueError, match="delay|support"):
            apply_sensor_delay(times, times[:, None], invalid, history, history[:, None])
    with pytest.raises(ValueError, match="continuously"):
        apply_sensor_delay(times, times[:, None], .024, history, np.ones((31, 1)))


def test_weak_uses_group1_and_same_group0_heldout(scenarios):
    weak = scenarios["weak"]
    record = weak.records["weak_step"]
    assert record.metadata["noise_group_id"] == 1
    assert record.metadata["noise"]["qpos"]["seed_sequence_entropy"] == (101, 1, 4, 0)
    assert record.metadata["excitation_diagnostics"]["active_torque_rms_nm"] == pytest.approx(.55)
    assert record.metadata["excitation_diagnostics"]["reset_count"] == 1
    np.testing.assert_array_equal(record.arrays["controls"][:50], 0)
    np.testing.assert_array_equal(record.arrays["controls"][50:-1], .55)
    assert record.arrays["controls"][-1, 0] == 0
    for name in weak.heldout:
        for key in ARRAY_NAMES:
            np.testing.assert_array_equal(weak.records[name].arrays[key], scenarios["matched"].records[name].arrays[key])
    assert sum(len(sequence.timestamps) for sequence in weak.training.sequences) == 8001


def test_mismatch_force_is_local_dissipative_and_absent_from_fitting_model(study, truth, scenarios, monkeypatch):
    model = compile_model(MODEL, truth.model)
    mass, inertia, damping = model.body_mass.copy(), model.body_inertia.copy(), model.dof_damping.copy()
    times = np.arange(101) * .002
    controls = np.ones((101, 1)) * .8
    before = simulate(model, times, controls)
    monkeypatch.setattr(mujoco, "set_mjcb_control", lambda *args: pytest.fail("must not install a global callback"))
    mismatch, extra = simulate_mismatch(model, times, controls, .08, .05)
    after = simulate(model, times, controls)
    for name in ("qpos", "qvel", "sensordata"):
        np.testing.assert_array_equal(getattr(before, name), getattr(after, name))
    np.testing.assert_array_equal(model.body_mass, mass)
    np.testing.assert_array_equal(model.body_inertia, inertia)
    np.testing.assert_array_equal(model.dof_damping, damping)
    np.testing.assert_array_equal(model.dof_frictionloss, 0)
    np.testing.assert_array_equal(extra, -.08 * np.tanh(mismatch.qvel / .05))
    assert np.all(extra * mismatch.qvel <= 0)
    assert np.max(np.abs(extra)) <= .08
    np.testing.assert_array_equal(mismatch.qvel[:2], before.qvel[:2])  # Friction at v0=0 is zero.
    assert mismatch.qvel[2, 0] < before.qvel[2, 0]  # Positive-velocity force opposes motion.
    for name, record in scenarios["mismatch"].records.items():
        assert not np.array_equal(record.arrays["clean_qpos"], scenarios["matched"].records[name].arrays["clean_qpos"])
        np.testing.assert_array_equal(record.arrays["nominal_qpos"], scenarios["matched"].records[name].arrays["nominal_qpos"])


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_versioned_scenario_roundtrip_and_output_preservation(tmp_path, scenarios, provenance, scenario):
    path = write_scenario(tmp_path / scenario, scenarios[scenario], provenance)
    loaded = load_scenario(path)
    assert loaded.metadata == scenarios[scenario].metadata
    assert loaded.scenario == scenario and loaded.seed == 101
    for name, record in loaded.records.items():
        for key in ARRAY_NAMES:
            np.testing.assert_array_equal(record.arrays[key], scenarios[scenario].records[name].arrays[key])
    before = sha256_file(path)
    with pytest.raises(ValueError, match="already exists"):
        write_scenario(path.parent, loaded, provenance)
    assert sha256_file(path) == before


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_training_loader_never_loads_heldout_or_latent_arrays(tmp_path, scenarios, provenance, scenario, monkeypatch):
    path = write_scenario(tmp_path / scenario, scenarios[scenario], provenance)
    for name in scenarios[scenario].heldout:
        (path.parent / f"{name}.npz").unlink()
    original_get = np.lib.npyio.NpzFile.__getitem__
    original_load = np.load
    opened = []; selected = []
    def load(file, *args, **kwargs):
        opened.append(Path(file).stem)
        assert kwargs.get("allow_pickle") is False
        return original_load(file, *args, **kwargs)
    def get(archive, key):
        selected.append(key)
        assert key in TRAIN_ARRAY_NAMES
        return original_get(archive, key)
    monkeypatch.setattr(np, "load", load)
    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", get)
    training = load_scenario_training(path)
    assert opened == list(scenarios[scenario].training_ids)
    assert selected == list(TRAIN_ARRAY_NAMES) * len(opened)
    assert tuple(sequence.trajectory_name for sequence in training.sequences) == scenarios[scenario].training_ids


def test_corrupt_hash_and_split_metadata_are_rejected(tmp_path, scenarios, provenance):
    path = write_scenario(tmp_path / "delayed", scenarios["delayed"], provenance)
    raw = json.loads(path.read_text())
    raw["trajectories"]["heldout_chirp"]["metadata"]["split"] = "train"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="split identity"):
        load_scenario_training(path)
    raw["trajectories"]["heldout_chirp"]["metadata"]["split"] = "heldout"
    path.write_text(json.dumps(raw))
    npz = path.parent / "train_multisine.npz"
    npz.write_bytes(npz.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_scenario_training(path)


def test_invalid_clean_transform_or_force_cannot_be_retained(scenarios):
    for scenario, key in (("delayed", "observed_clean_qpos"), ("matched", "extra_torque_nm")):
        data = scenarios[scenario]
        name = data.training_ids[0]
        arrays = dict(data.records[name].arrays)
        arrays[key] = arrays[key] + .1
        records = dict(data.records)
        records[name] = TrajectoryRecord(arrays, data.records[name].metadata)
        with pytest.raises(ValueError, match="noise|transform|force"):
            validate_scenario(replace(data, records=records))


@pytest.mark.parametrize("scenario,seed", [("unsupported", 101), ("matched", 999), ("matched", True)])
def test_invalid_scenario_or_seed_rejected(study, truth, scenario, seed):
    with pytest.raises(ValueError):
        create_scenario(study, truth, MODEL, seed, scenario)
