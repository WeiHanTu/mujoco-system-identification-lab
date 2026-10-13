from dataclasses import fields, replace
import json
from pathlib import Path

import numpy as np
import pytest

from mujoco_sysid_lab.dataset import (
    ARRAY_NAMES, CORE_TRAJECTORIES, TRAIN_ARRAY_NAMES, Dataset, TrainingDataset,
    TrainingSequence, TrajectoryRecord, array_sha256, generate_dataset,
    load_evaluation_dataset, load_training_sequences, validate_dataset, write_dataset,
)
from mujoco_sysid_lab.provenance import collect_provenance, sha256_file
from mujoco_sysid_lab.experiment_config import load_ground_truth, load_study_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "model" / "pendulum.xml"


@pytest.fixture(scope="module")
def study():
    return load_study_config(PROJECT_ROOT / "configs" / "evidence_v1.json")


@pytest.fixture(scope="module")
def truth(study):
    return load_ground_truth(study.truth_config_path, study)


@pytest.fixture(scope="module")
def dataset(study, truth):
    return generate_dataset(study, truth, MODEL_PATH, 101)


@pytest.fixture(scope="module")
def provenance():
    # Actual HEAD/library/environment record; these are ephemeral test files.
    return collect_provenance(
        PROJECT_ROOT, PROJECT_ROOT / "configs" / "evidence_v1.json", MODEL_PATH,
        [".venv/bin/python", "-m", "pytest", "tests/test_dataset.py"],
    )


@pytest.fixture
def written(tmp_path, dataset, provenance):
    return write_dataset(tmp_path / "dataset", dataset, provenance)


def _changed_record(dataset, name, *, arrays=None, metadata=None):
    record = dataset.records[name]
    records = dict(dataset.records)
    records[name] = TrajectoryRecord(arrays if arrays is not None else record.arrays,
                                     metadata if metadata is not None else record.metadata)
    return replace(dataset, records=records)


def _edit_manifest(path, change):
    raw = json.loads(path.read_text())
    change(raw)
    path.write_text(json.dumps(raw, allow_nan=False))


def test_generation_repeats_every_array_and_metadata_exactly(study, truth, dataset):
    repeat = generate_dataset(study, truth, MODEL_PATH, 101)
    assert repeat.metadata == dataset.metadata
    for name in CORE_TRAJECTORIES:
        assert repeat.records[name].metadata == dataset.records[name].metadata
        for key in ARRAY_NAMES:
            actual = repeat.records[name].arrays[key]
            np.testing.assert_array_equal(actual, dataset.records[name].arrays[key])
            assert not actual.flags.writeable


def test_observation_seed_changes_only_noise_not_controls_or_physics(study, truth, dataset):
    other = generate_dataset(study, truth, MODEL_PATH, 102)
    for name in CORE_TRAJECTORIES:
        a, b = dataset.records[name].arrays, other.records[name].arrays
        for key in ARRAY_NAMES - {"measured_qpos", "measured_qvel", "innovation_qpos", "innovation_qvel"}:
            np.testing.assert_array_equal(a[key], b[key])
        for key in ("measured_qpos", "measured_qvel", "innovation_qpos", "innovation_qvel"):
            assert not np.array_equal(a[key], b[key])


def test_registered_shapes_grid_initialization_prehistory_and_split(dataset):
    assert dataset.scenario == "matched" and dataset.seed == 101
    assert tuple(dataset.records) == CORE_TRAJECTORIES
    for trajectory_id, name in enumerate(CORE_TRAJECTORIES):
        record = dataset.records[name]
        assert record.metadata["trajectory_id"] == trajectory_id
        assert record.metadata["split"] == ("train" if trajectory_id < 2 else "heldout")
        assert set(record.arrays) == ARRAY_NAMES
        np.testing.assert_array_equal(record.arrays["timestamps"], np.arange(4001) * 0.002)
        np.testing.assert_array_equal(record.arrays["initial_state"], [0.0, 0.0, 0.0])
        assert record.arrays["timestamps"].shape == (4001,)
        for key in ARRAY_NAMES - {"timestamps", "initial_state"}:
            if not key.startswith("prehistory_"):
                assert record.arrays[key].shape == (4001, 1)
        history = record.arrays["prehistory_timestamps"]
        assert history.shape == (31,) and history[0] == -0.060 and history[-1] == 0
        np.testing.assert_allclose(np.diff(history), 0.002, rtol=0, atol=1e-17)
        np.testing.assert_array_equal(record.arrays["prehistory_initial_state"], [-0.060, 0, 0])
        for key in ("prehistory_controls", "prehistory_qpos", "prehistory_qvel"):
            np.testing.assert_array_equal(record.arrays[key], np.zeros((31, 1)))
        for array in record.arrays.values():
            assert array.dtype == np.float64 and np.isfinite(array).all()
            assert not array.flags.writeable


def test_noise_keys_and_scales_match_registered_independent_streams(dataset):
    streams = []
    for name, record in dataset.records.items():
        for sensor_id, (sensor, sigma) in enumerate((("qpos", 0.002), ("qvel", 0.020))):
            metadata = record.metadata["noise"][sensor]
            entropy = [101, 0, record.metadata["trajectory_id"], sensor_id]
            assert metadata["seed_sequence_entropy"] == entropy
            expected = np.random.Generator(np.random.PCG64(
                np.random.SeedSequence(entropy),
            )).standard_normal((4001, 1))
            actual = record.arrays[f"innovation_{sensor}"]
            np.testing.assert_array_equal(actual, expected)
            assert metadata["innovation_sha256"] == array_sha256(actual)
            np.testing.assert_array_equal(record.arrays[f"measured_{sensor}"],
                                          record.arrays[f"clean_{sensor}"] + sigma * actual)
            assert np.std(actual) == pytest.approx(1.0, rel=0.06)
            assert abs(np.mean(actual)) <= 4 / np.sqrt(actual.size)
            streams.append(actual)
    for a, b in zip(streams[:-1], streams[1:], strict=True):
        assert not np.array_equal(a, b)
        assert abs(np.corrcoef(a[:, 0], b[:, 0])[0, 1]) < 0.08


def test_ground_truth_and_nominal_latent_trajectories_differ(dataset, truth, study):
    assert (truth.model.mass_kg, study.nominal.mass_kg) == (1.2, 0.8)
    assert (truth.model.damping_nm_s_per_rad, study.nominal.damping_nm_s_per_rad) == (0.15, 0.04)
    for record in dataset.records.values():
        assert not np.array_equal(record.arrays["clean_qpos"], record.arrays["nominal_qpos"])
        assert not np.array_equal(record.arrays["clean_qvel"], record.arrays["nominal_qvel"])
    assert dataset.metadata["truth_config_sha256"] == sha256_file(study.truth_config_path)
    assert dataset.metadata["model_source_sha256"] == sha256_file(MODEL_PATH)


@pytest.mark.parametrize("seed", [-1, True, 1.5, 0, 999])
def test_unregistered_or_invalid_noise_seed_is_rejected(study, truth, seed):
    with pytest.raises(ValueError, match="seed"):
        generate_dataset(study, truth, MODEL_PATH, seed)


def test_nonzero_delay_is_not_silently_generated_in_milestone2(study, truth):
    with pytest.raises(ValueError, match="zero delay"):
        generate_dataset(study, replace(truth, matched_delay_s=0.024), MODEL_PATH, 101)


def test_full_roundtrip_verifies_all_data_and_strict_manifest(written, dataset):
    loaded = load_evaluation_dataset(written)
    assert loaded.metadata == dataset.metadata
    assert loaded.seed == dataset.seed and loaded.scenario == dataset.scenario
    manifest = json.loads(written.read_text())
    assert manifest["dataset_schema_version"] == 1
    assert manifest["provenance"]["command"]["argv"] == [
        ".venv/bin/python", "-m", "pytest", "tests/test_dataset.py",
    ]
    for name in CORE_TRAJECTORIES:
        entry = manifest["trajectories"][name]
        assert entry["file_sha256"] == sha256_file(written.parent / entry["file"])
        assert loaded.records[name].metadata == dataset.records[name].metadata
        for key in ARRAY_NAMES:
            actual = loaded.records[name].arrays[key]
            np.testing.assert_array_equal(actual, dataset.records[name].arrays[key])
            assert entry["arrays"][key]["sha256"] == array_sha256(actual)
            assert not actual.flags.writeable


def test_training_reader_does_not_open_heldout_files_or_load_hidden_arrays(written, monkeypatch):
    # Deliberately remove evaluation files. Training isolation remains usable;
    # full dataset verification separately requires those files to be present.
    for name in CORE_TRAJECTORIES[2:]:
        (written.parent / f"{name}.npz").unlink()
    original_load = np.load
    original_getitem = np.lib.npyio.NpzFile.__getitem__
    opened = []
    selected = []

    def observed_load(path, *args, **kwargs):
        opened.append(Path(path).name)
        assert kwargs.get("allow_pickle") is False
        return original_load(path, *args, **kwargs)

    def observed_getitem(archive, key):
        selected.append(key)
        assert key in TRAIN_ARRAY_NAMES
        return original_getitem(archive, key)

    monkeypatch.setattr(np, "load", observed_load)
    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", observed_getitem)
    training = load_training_sequences(written.parent)
    assert isinstance(training, TrainingDataset)
    assert opened == ["train_multisine.npz", "train_prbs.npz"]
    assert selected == list(TRAIN_ARRAY_NAMES) * 2
    assert {field.name for field in fields(TrainingDataset)} == {
        "seed", "sequences", "dataset_schema_version",
    }
    assert {field.name for field in fields(TrainingSequence)} == {
        "trajectory_name", "trajectory_id", *TRAIN_ARRAY_NAMES,
    }
    assert tuple(sequence.trajectory_name for sequence in training.sequences) == CORE_TRAJECTORIES[:2]
    for sequence in training.sequences:
        assert sequence.initial_state.shape == (3,)
        assert not sequence.timestamps.flags.writeable
        assert not hasattr(sequence, "clean_qpos") and not hasattr(sequence, "metadata")


@pytest.mark.parametrize("name,split", [("heldout_chirp", "train"), ("train_prbs", "heldout")])
def test_manifest_split_relabeling_is_rejected_before_arrays_read(written, monkeypatch, name, split):
    _edit_manifest(written, lambda raw: raw["trajectories"][name]["metadata"].update(split=split))
    monkeypatch.setattr(np, "load", lambda *args, **kwargs: pytest.fail("must reject split before loading arrays"))
    with pytest.raises(ValueError, match="split integrity"):
        load_training_sequences(written)


def test_corrupted_file_hash_is_rejected(written):
    path = written.parent / "train_multisine.npz"
    path.write_bytes(path.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="file hash mismatch"):
        load_training_sequences(written)


def test_altered_array_with_recomputed_file_hash_is_rejected(written):
    path = written.parent / "train_multisine.npz"
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["measured_qpos"][20, 0] += 0.1
    np.savez_compressed(path, **arrays)
    _edit_manifest(written, lambda raw: raw["trajectories"]["train_multisine"].update(
        file_sha256=sha256_file(path)))
    with pytest.raises(ValueError, match="array hash/shape/dtype mismatch"):
        load_evaluation_dataset(written)


@pytest.mark.parametrize("edit,error", [
    (lambda raw: raw.update(dataset_schema_version=2), "unsupported dataset_schema_version"),
    (lambda raw: raw.update(dataset_schema_version=True), "unsupported dataset_schema_version"),
    (lambda raw: raw.update(extra=True), "versioned schema"),
    (lambda raw: raw["trajectories"].pop("heldout_pulse"), "four registered"),
    (lambda raw: raw["trajectories"]["train_prbs"]["metadata"].update(trajectory_id=0), "trajectory ID"),
    (lambda raw: raw["trajectories"]["train_prbs"].update(file="../outside.npz"), "relative filename"),
    (lambda raw: raw.update(metadata="invalid"), "metadata must be an object"),
])
def test_invalid_manifest_schema_is_rejected(written, edit, error):
    _edit_manifest(written, edit)
    with pytest.raises(ValueError, match=error):
        load_evaluation_dataset(written)


@pytest.mark.parametrize("contents,error", [
    ('{"seed": 101, "seed": 102}', "duplicate JSON field"),
    ('{"seed": NaN}', "non-finite JSON constant"),
    ('{"seed": 1e400}', "finite JSON-compatible"),
])
def test_nonfinite_or_duplicate_manifest_json_is_rejected(tmp_path, contents, error):
    path = tmp_path / "manifest.json"
    path.write_text(contents)
    with pytest.raises(ValueError, match=error):
        load_training_sequences(path)


def test_writer_preserves_existing_evidence_and_rejects_invalid_json_before_creating(tmp_path, dataset, provenance):
    output = tmp_path / "new_dataset"
    write_dataset(output, dataset, provenance)
    before = {file.name: sha256_file(file) for file in output.iterdir()}
    with pytest.raises(ValueError, match="Output already exists"):
        write_dataset(output, dataset, provenance)
    assert {file.name: sha256_file(file) for file in output.iterdir()} == before
    invalid_output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="finite JSON-compatible"):
        write_dataset(invalid_output, dataset, {"runtime": float("nan")})
    assert not invalid_output.exists()


@pytest.mark.parametrize("missing", [
    "mujoco_version", "python", "environment", "git", "command", "config", "model", "package_versions",
])
def test_writer_rejects_missing_provenance_fields_before_creation(tmp_path, dataset, provenance, missing):
    incomplete = dict(provenance)
    del incomplete[missing]
    with pytest.raises(ValueError, match="provenance missing required"):
        write_dataset(tmp_path / "incomplete", dataset, incomplete)
    assert not (tmp_path / "incomplete").exists()


def test_writer_rejects_empty_provenance_and_mismatched_model_or_seed(tmp_path, dataset, provenance):
    with pytest.raises(ValueError, match="provenance missing required"):
        write_dataset(tmp_path / "empty", dataset, {})
    wrong_model = dict(provenance, model={"sha256": "0" * 64})
    with pytest.raises(ValueError, match="model SHA256"):
        write_dataset(tmp_path / "wrong_model", dataset, wrong_model)
    with pytest.raises(ValueError, match="seed differs"):
        write_dataset(tmp_path / "wrong_seed", dataset, dict(provenance, seed=102))
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize("key", sorted(ARRAY_NAMES))
def test_nonfinite_generated_array_cannot_be_retained(dataset, tmp_path, key):
    arrays = dict(dataset.records["train_multisine"].arrays)
    bad = arrays[key].copy()
    bad.flat[0] = np.nan
    arrays[key] = bad
    with pytest.raises(ValueError, match="finite float64"):
        write_dataset(tmp_path / "bad", _changed_record(dataset, "train_multisine", arrays=arrays), {})
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("key,replacement,error", [
    ("timestamps", np.arange(4001) * 0.003, "fixed strictly increasing grid"),
    ("timestamps", np.zeros(4001), "fixed strictly increasing grid"),
    ("controls", np.zeros(4001), "shape.*N, 1"),
    ("measured_qvel", np.zeros((4000, 1)), "shape.*N, 1"),
    ("initial_state", np.zeros(2), "FULLPHYSICS"),
    ("initial_state", np.array([0.0, 0.1, 0.2]), "FULLPHYSICS"),
    ("prehistory_qpos", np.ones((31, 1)), "zero-control equilibrium"),
])
def test_shapes_timing_state_and_prehistory_corruption_are_rejected(dataset, key, replacement, error):
    arrays = dict(dataset.records["train_multisine"].arrays)
    arrays[key] = replacement
    with pytest.raises(ValueError, match=error):
        validate_dataset(_changed_record(dataset, "train_multisine", arrays=arrays))


def test_array_hash_distinguishes_shape_and_dtype():
    data = np.arange(6, dtype=np.float64)
    assert array_sha256(data) != array_sha256(data.reshape(2, 3))
    assert array_sha256(data) != array_sha256(data.view(np.uint64))
    assert array_sha256(data) == array_sha256(data.copy())
