"""Versioned synthetic datasets and a deliberately narrow training reader.

Only :func:`load_training_sequences` is an estimator input interface.  Truth,
latent signals, nominal predictions, and held-out arrays are evaluation data.
The training reader never opens a held-out NPZ, including during verification.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, TYPE_CHECKING

import mujoco
import numpy as np
from numpy.typing import NDArray

from mujoco_sysid_lab.provenance import sha256_file, write_json
from mujoco_sysid_lab.simulation import compile_model, simulate

if TYPE_CHECKING:
    from mujoco_sysid_lab.experiment_config import StudyConfig, TruthConfig


DATASET_SCHEMA_VERSION = 1
CORE_TRAJECTORIES = (
    "train_multisine", "train_prbs", "heldout_chirp", "heldout_pulse",
)
TRAJECTORY_IDS = dict(zip(CORE_TRAJECTORIES, range(4), strict=True))
TRAIN_TRAJECTORIES = CORE_TRAJECTORIES[:2]
ARRAY_NAMES = frozenset({
    "timestamps", "controls", "measured_qpos", "measured_qvel",
    "clean_qpos", "clean_qvel", "nominal_qpos", "nominal_qvel",
    "initial_state", "innovation_qpos", "innovation_qvel",
    "prehistory_timestamps", "prehistory_controls", "prehistory_qpos",
    "prehistory_qvel", "prehistory_initial_state",
})
TRAIN_ARRAY_NAMES = (
    "timestamps", "controls", "measured_qpos", "measured_qvel", "initial_state",
)
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class TrajectoryRecord:
    arrays: Mapping[str, FloatArray]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class Dataset:
    seed: int
    scenario: str
    records: Mapping[str, TrajectoryRecord]
    metadata: Mapping[str, Any]
    dataset_schema_version: int = DATASET_SCHEMA_VERSION


@dataclass(frozen=True)
class TrainingSequence:
    """Known training inputs only; intentionally has no metadata/truth field."""

    trajectory_name: str
    trajectory_id: int
    timestamps: FloatArray
    controls: FloatArray
    measured_qpos: FloatArray
    measured_qvel: FloatArray
    initial_state: FloatArray


@dataclass(frozen=True)
class TrainingDataset:
    seed: int
    sequences: tuple[TrainingSequence, ...]
    dataset_schema_version: int = DATASET_SCHEMA_VERSION


def array_sha256(array: NDArray) -> str:
    """Hash dtype, shape, and C-order bytes, rather than an ambiguous byte blob."""
    contiguous = np.ascontiguousarray(array)
    descriptor = json.dumps(
        {"dtype": contiguous.dtype.str, "shape": list(contiguous.shape)},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(descriptor + b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _snapshot(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _snapshot(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _snapshot(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_snapshot(item) for item in value]
    if isinstance(value, Path):
        # Input file hashes/provenance record the full repository-relative path.
        # A snapshot must not leak a developer's home path into public metadata.
        return value.name
    if isinstance(value, np.generic):
        return value.item()
    return value


def _owned_readonly(array: NDArray) -> FloatArray:
    result = np.array(array, dtype=np.float64, copy=True, order="C")
    result.flags.writeable = False
    return result


def _initial_state(model: mujoco.MjModel, qpos: float, qvel: float) -> FloatArray:
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.time = 0.0
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    mujoco.mj_forward(model, data)
    state_spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
    state = np.empty(mujoco.mj_stateSize(model, state_spec), dtype=np.float64)
    mujoco.mj_getState(model, data, state, state_spec)
    if state.shape != (3,):
        raise ValueError("dataset requires FULLPHYSICS [time, qpos, qvel] with na=0")
    return _owned_readonly(state)


def _strict_json_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    try:
        serialized = json.dumps(_snapshot(value), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must contain finite JSON-compatible values") from error
    return json.loads(serialized)


def _validate_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("dataset seed must be a nonnegative integer")


def generate_dataset(study: StudyConfig, truth: TruthConfig, model_path: Path,
                     seed: int) -> Dataset:
    """Generate the four complete matched-model trajectories for a fixed seed.

    Noise follows PCG64(SeedSequence([seed, 0, trajectory_id, sensor_id])).
    Controls and clean dynamics are independent of observation-noise seed.
    Delayed/misspecified generators belong to their later milestone gates.
    """
    from mujoco_sysid_lab.excitation import generate_control

    _validate_seed(seed)
    if seed not in study.noise_seeds:
        raise ValueError("dataset seed is outside the registered noise seeds")
    if truth.matched_delay_s != 0:
        raise ValueError("Milestone 2 requires zero delay in the matched scenario")
    if (truth.model.initial_qpos_rad, truth.model.initial_qvel_rad_s) != (0.0, 0.0):
        raise ValueError("registered prehistory requires zero equilibrium initialization")
    dt = study.nominal.timestep_s
    intervals = study.prehistory_s / dt
    if (intervals < 1 or not np.isclose(intervals, round(intervals), rtol=0, atol=1e-9)):
        raise ValueError("prehistory_s must be a positive integer multiple of timestep_s")
    target_model = compile_model(model_path, truth.model)
    nominal_model = compile_model(model_path, study.nominal)
    state = _initial_state(target_model, truth.model.initial_qpos_rad,
                           truth.model.initial_qvel_rad_s)
    nominal_state = _initial_state(nominal_model, study.nominal.initial_qpos_rad,
                                   study.nominal.initial_qvel_rad_s)
    if not np.array_equal(state, nominal_state):
        raise ValueError("truth and nominal models must use the same known initial state")
    history_clock = np.arange(round(intervals) + 1, dtype=np.float64) * dt
    history = simulate(target_model, history_clock, np.zeros((history_clock.size, 1)),
                       initial_qpos=truth.model.initial_qpos_rad,
                       initial_qvel=truth.model.initial_qvel_rad_s)
    history_times = history_clock - study.prehistory_s
    history_times[-1] = 0.0
    history_state = np.array(state, copy=True)
    history_state[0] = -study.prehistory_s
    records: dict[str, TrajectoryRecord] = {}
    for name in CORE_TRAJECTORIES:
        excitation = study.trajectories[name]
        if excitation.trajectory_id != TRAJECTORY_IDS[name]:
            raise ValueError(f"unregistered trajectory ID for {name}")
        expected_split = "train" if name in TRAIN_TRAJECTORIES else "heldout"
        if excitation.split != expected_split:
            raise ValueError(f"split integrity failure for {name}")
        times, controls, diagnostics = generate_control(study, name)
        clean = simulate(target_model, times, controls,
                         initial_qpos=truth.model.initial_qpos_rad,
                         initial_qvel=truth.model.initial_qvel_rad_s)
        nominal = simulate(nominal_model, times, controls,
                           initial_qpos=study.nominal.initial_qpos_rad,
                           initial_qvel=study.nominal.initial_qvel_rad_s)
        innovations: dict[str, FloatArray] = {}
        noise_metadata: dict[str, Any] = {}
        for sensor_id, sensor in enumerate(("qpos", "qvel")):
            entropy = [seed, 0, excitation.trajectory_id, sensor_id]
            innovation = np.random.Generator(np.random.PCG64(
                np.random.SeedSequence(entropy),
            )).standard_normal((times.size, 1))
            innovations[sensor] = innovation
            noise_metadata[sensor] = {
                "sigma": study.sensor_sigmas[sensor],
                "seed_sequence_entropy": entropy,
                "innovation_sha256": array_sha256(innovation),
            }
        arrays = {
            "timestamps": times, "controls": controls,
            "measured_qpos": clean.qpos + study.sensor_sigmas["qpos"] * innovations["qpos"],
            "measured_qvel": clean.qvel + study.sensor_sigmas["qvel"] * innovations["qvel"],
            "clean_qpos": clean.qpos, "clean_qvel": clean.qvel,
            "nominal_qpos": nominal.qpos, "nominal_qvel": nominal.qvel,
            "initial_state": state, "innovation_qpos": innovations["qpos"],
            "innovation_qvel": innovations["qvel"],
            "prehistory_timestamps": history_times,
            "prehistory_controls": history.controls,
            "prehistory_qpos": history.qpos, "prehistory_qvel": history.qvel,
            "prehistory_initial_state": history_state,
        }
        metadata = {
            "trajectory_id": excitation.trajectory_id, "split": expected_split,
            "duration_s": excitation.duration_s, "timestep_s": dt,
            "excitation_config": _snapshot(excitation),
            "excitation_diagnostics": _snapshot(diagnostics),
            "noise": noise_metadata, "prehistory_s": study.prehistory_s,
            "sensor_delay_s": 0.0,
            "initial_state_spec": "mjSTATE_FULLPHYSICS",
            "initial_state_order": ["time_s", "qpos_rad", "qvel_rad_s"],
            "initial_state_source": "known exact reset, not measured/refitted",
        }
        records[name] = TrajectoryRecord(
            MappingProxyType({key: _owned_readonly(value) for key, value in arrays.items()}),
            MappingProxyType(metadata),
        )
    dataset = Dataset(seed=seed, scenario="matched", records=MappingProxyType(records),
                      metadata=MappingProxyType({
                          "study_config": _snapshot(study),
                          "truth_generator_config": _snapshot(truth),
                          "truth_config_sha256": sha256_file(study.truth_config_path),
                          "model_source_sha256": sha256_file(model_path),
                          "units": {"timestamps": "s", "controls": "motor control units",
                                    "qpos": "rad", "qvel": "rad/s"},
                          "control_semantics": "u[k] drives [t[k], t[k+1]); u[-1] unused",
                          "sensor_semantics": "fresh forward sensors at state time t[k]",
                          "noise_scheme": "PCG64 SeedSequence([seed, group, trajectory_id, sensor_id])",
                          "noise_group_id": 0,
                          "noise_distribution": "independent zero-mean Gaussian observation noise",
                          "noise_order": "additive measurement noise after clean signal/delay",
                          "process_noise": "none",
                          "split_unit": "complete trajectory ID",
                          "prehistory_generation": (
                              "explicit zero-control equilibrium simulation on local clock [0,H]; "
                              "autonomous dynamics relabeled to [-H,0]"
                          ),
                      }))
    validate_dataset(dataset)
    return dataset


def _validate_timed_arrays(arrays: Mapping[str, NDArray], metadata: Mapping[str, Any],
                           *, full: bool) -> None:
    expected_names = ARRAY_NAMES if full else frozenset(TRAIN_ARRAY_NAMES)
    if set(arrays) != expected_names:
        raise ValueError("trajectory arrays do not match the versioned schema")
    for name, array in arrays.items():
        if (not isinstance(array, np.ndarray) or array.dtype != np.dtype(np.float64)
                or not np.isfinite(array).all()):
            raise ValueError(f"{name} must be a finite float64 array")
    times = arrays["timestamps"]
    dt, duration = metadata["timestep_s"], metadata["duration_s"]
    if (isinstance(dt, bool) or not isinstance(dt, (int, float)) or not np.isfinite(dt)
            or dt <= 0 or isinstance(duration, bool) or not isinstance(duration, (int, float))
            or not np.isfinite(duration) or duration <= 0):
        raise ValueError("trajectory duration/timestep must be finite and positive")
    if times.ndim != 1 or times.size < 2:
        raise ValueError("timestamps must be one-dimensional with at least two samples")
    if (not np.all(np.diff(times) > 0) or times[0] != 0.0
            or not np.allclose(times, np.arange(times.size) * dt, rtol=0, atol=1e-12)
            or not np.isclose(times[-1], duration, rtol=0, atol=1e-12)):
        raise ValueError("timestamps must span [0,duration] on the fixed strictly increasing grid")
    for name in (set(arrays) - {"timestamps", "initial_state"}):
        if not name.startswith("prehistory_") and arrays[name].shape != (times.size, 1):
            raise ValueError(f"{name} must have shape (N, 1)")
    initial_state = arrays["initial_state"]
    if initial_state.shape != (3,) or not np.array_equal(initial_state, [0.0, 0.0, 0.0]):
        raise ValueError("matched initial_state must be FULLPHYSICS [time=0, qpos=0, qvel=0]")
    if full:
        for sensor, state_index in (("qpos", 1), ("qvel", 2)):
            if (arrays[f"clean_{sensor}"][0, 0] != initial_state[state_index]
                    or arrays[f"nominal_{sensor}"][0, 0] != initial_state[state_index]):
                raise ValueError("all baselines must share the explicit known initial state")
        pre_times = arrays["prehistory_timestamps"]
        prehistory_s = metadata["prehistory_s"]
        if (not isinstance(prehistory_s, (int, float)) or isinstance(prehistory_s, bool)
                or not np.isfinite(prehistory_s) or prehistory_s <= 0):
            raise ValueError("prehistory_s must be finite and positive")
        if (pre_times.ndim != 1 or pre_times.size < 2 or pre_times[-1] != 0
                or not np.all(np.diff(pre_times) > 0)
                or not np.allclose(pre_times, np.arange(pre_times.size) * dt - prehistory_s,
                                   rtol=0, atol=1e-12)):
            raise ValueError("prehistory timestamps must span [-prehistory_s,0] on the fixed grid")
        for name in ("prehistory_controls", "prehistory_qpos", "prehistory_qvel"):
            if arrays[name].shape != (pre_times.size, 1):
                raise ValueError(f"{name} must have shape (H,1)")
            if np.any(arrays[name] != 0.0):
                raise ValueError("matched prehistory must be a zero-control equilibrium")
        if not np.array_equal(arrays["prehistory_initial_state"], [-prehistory_s, 0.0, 0.0]):
            raise ValueError("prehistory_initial_state must record explicit equilibrium at negative time")
        if metadata.get("sensor_delay_s") != 0.0:
            raise ValueError("matched Milestone 2 data must have zero sensor delay")


def validate_dataset(dataset: Dataset) -> None:
    if (type(dataset.dataset_schema_version) is not int
            or dataset.dataset_schema_version != DATASET_SCHEMA_VERSION):
        raise ValueError("unsupported dataset_schema_version")
    _validate_seed(dataset.seed)
    if dataset.scenario != "matched":
        raise ValueError("only the matched Milestone 2 scenario is supported")
    if set(dataset.records) != set(CORE_TRAJECTORIES):
        raise ValueError("dataset requires exactly four registered complete trajectories")
    _strict_json_object(dataset.metadata, "dataset metadata")
    for name, record in dataset.records.items():
        metadata = _strict_json_object(record.metadata, f"{name} metadata")
        _validate_record_identity(name, metadata)
        _validate_timed_arrays(record.arrays, metadata, full=True)
        for sensor_id, sensor in enumerate(("qpos", "qvel")):
            noise = metadata["noise"][sensor]
            expected_entropy = [dataset.seed, 0, TRAJECTORY_IDS[name], sensor_id]
            if noise.get("seed_sequence_entropy") != expected_entropy:
                raise ValueError("noise seed keys disagree with the registered dataset identity")
            sigma = noise["sigma"]
            if (isinstance(sigma, bool) or not isinstance(sigma, (int, float))
                    or not np.isfinite(sigma) or sigma <= 0):
                raise ValueError("noise sigma must be finite and positive")
            innovation = record.arrays[f"innovation_{sensor}"]
            if array_sha256(innovation) != noise.get("innovation_sha256"):
                raise ValueError("innovation hash mismatch")
            if not np.array_equal(record.arrays[f"measured_{sensor}"],
                                  record.arrays[f"clean_{sensor}"] + sigma * innovation):
                raise ValueError("measured data must equal clean data plus registered observation noise")


def _validate_record_identity(name: str, metadata: Mapping[str, Any]) -> None:
    if (name not in TRAJECTORY_IDS or type(metadata.get("trajectory_id")) is not int
            or metadata.get("trajectory_id") != TRAJECTORY_IDS[name]):
        raise ValueError("unknown, duplicate, or incorrect trajectory ID")
    expected_split = "train" if name in TRAIN_TRAJECTORIES else "heldout"
    if metadata.get("split") != expected_split:
        raise ValueError(f"split integrity failure for {name}")
    for required in ("timestep_s", "duration_s", "prehistory_s", "noise"):
        if required not in metadata:
            raise ValueError(f"trajectory metadata missing required {required}")
    noise = metadata["noise"]
    if not isinstance(noise, dict) or set(noise) != {"qpos", "qvel"}:
        raise ValueError("noise metadata must describe both registered sensors")
    for sensor in ("qpos", "qvel"):
        descriptor = noise[sensor]
        if (not isinstance(descriptor, dict)
                or set(descriptor) != {"sigma", "seed_sequence_entropy", "innovation_sha256"}
                or not _valid_sha256(descriptor["innovation_sha256"])):
            raise ValueError("invalid sensor noise descriptor")
        sigma = descriptor["sigma"]
        if (isinstance(sigma, bool) or not isinstance(sigma, (int, float))
                or not np.isfinite(sigma) or sigma <= 0):
            raise ValueError("noise sigma must be finite and positive")


def write_dataset(output: Path, dataset: Dataset, provenance: Mapping[str, Any]) -> Path:
    """Write new per-trajectory NPZ files plus a strict, hashed JSON manifest.

    Existing directories are never reused.  Invalid input is rejected before a
    directory is created; unexpected IO failures leave partial evidence intact.
    Caller supplies the already collected exact environment/source provenance.
    """
    validate_dataset(dataset)
    provenance_snapshot = _validate_provenance(provenance, dataset)
    metadata_snapshot = _strict_json_object(dataset.metadata, "dataset metadata")
    target = Path(output)
    if target.exists():
        raise ValueError("Output already exists; choose a new run directory to preserve evidence")
    target.mkdir(parents=True, exist_ok=False)
    entries: dict[str, Any] = {}
    for name in CORE_TRAJECTORIES:
        record = dataset.records[name]
        npz_path = target / f"{name}.npz"
        np.savez_compressed(npz_path, **record.arrays)
        entries[name] = {
            "file": npz_path.name, "file_sha256": sha256_file(npz_path),
            "metadata": _strict_json_object(record.metadata, f"{name} metadata"),
            "arrays": {
                key: {"shape": list(array.shape), "dtype": array.dtype.str,
                      "sha256": array_sha256(array)}
                for key, array in sorted(record.arrays.items())
            },
        }
    manifest = {
        "dataset_schema_version": dataset.dataset_schema_version,
        "seed": dataset.seed, "scenario": dataset.scenario,
        "metadata": metadata_snapshot, "provenance": provenance_snapshot,
        "trajectories": entries,
    }
    path = target / "manifest.json"
    write_json(path, manifest)
    return path


def _validate_provenance(provenance: Mapping[str, Any], dataset: Dataset) -> dict[str, Any]:
    record = _strict_json_object(provenance, "provenance")
    required = {"mujoco_version", "python", "environment", "git", "command",
                "config", "model", "package_versions"}
    missing = sorted(required - set(record))
    if missing:
        raise ValueError(f"dataset provenance missing required fields: {missing}")
    if record["mujoco_version"] != mujoco.__version__:
        raise ValueError("dataset provenance MuJoCo version differs from the active library")
    for name in ("python", "environment", "git", "command", "config", "model", "package_versions"):
        if not isinstance(record[name], dict):
            raise ValueError(f"dataset provenance {name} must be an object")
    if not isinstance(record["python"].get("version"), str) or not record["python"]["version"]:
        raise ValueError("dataset provenance requires Python version")
    environment = record["environment"]
    if (not isinstance(environment.get("cpu"), dict) or not environment["cpu"]
            or not isinstance(environment.get("os"), dict) or not environment["os"]):
        raise ValueError("dataset provenance requires OS and CPU records")
    git = record["git"]
    commit = git.get("commit")
    if (not isinstance(commit, str) or len(commit) not in (40, 64)
            or any(character not in "0123456789abcdef" for character in commit)
            or type(git.get("dirty")) is not bool):
        raise ValueError("dataset provenance requires Git commit and dirty/clean state")
    argv = record["command"].get("argv")
    if (not isinstance(argv, list) or not argv
            or any(not isinstance(argument, str) or not argument for argument in argv)):
        raise ValueError("dataset provenance requires exact command argv")
    config = record["config"]
    if not isinstance(config.get("content"), dict) or not _valid_sha256(config.get("sha256")):
        raise ValueError("dataset provenance requires exact config content and SHA256")
    if (not _valid_sha256(record["model"].get("sha256"))
            or record["model"]["sha256"] != dataset.metadata.get("model_source_sha256")):
        raise ValueError("dataset provenance model SHA256 disagrees with the generator")
    packages = record["package_versions"]
    if (packages.get("mujoco") != record["mujoco_version"]
            or any(not isinstance(name, str) or not isinstance(version, str) or not version
                   for name, version in packages.items())):
        raise ValueError("dataset provenance requires exact installed package versions")
    # The dataset's seed is mandatory and validated separately; an optional
    # duplicated provenance seed must agree rather than silently override it.
    if "seed" in record and record["seed"] != dataset.seed:
        raise ValueError("dataset provenance seed differs from the dataset")
    return record


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value} is forbidden")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _read_manifest(path: Path) -> tuple[Path, dict[str, Any]]:
    manifest_path = Path(path)
    if manifest_path.is_dir():
        manifest_path = manifest_path / "manifest.json"
    if manifest_path.is_symlink():
        raise ValueError("dataset manifest must be a regular file")
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream, parse_constant=_reject_constant,
                             object_pairs_hook=_unique_object)
    manifest = _strict_json_object(manifest, "manifest")
    if set(manifest) != {"dataset_schema_version", "seed", "scenario", "metadata",
                         "provenance", "trajectories"}:
        raise ValueError("dataset manifest does not match the versioned schema")
    if (type(manifest["dataset_schema_version"]) is not int
            or manifest["dataset_schema_version"] != DATASET_SCHEMA_VERSION):
        raise ValueError("unsupported dataset_schema_version")
    _validate_seed(manifest["seed"])
    if manifest["scenario"] != "matched":
        raise ValueError("unsupported dataset scenario")
    _strict_json_object(manifest["metadata"], "dataset metadata")
    _strict_json_object(manifest["provenance"], "provenance")
    entries = manifest["trajectories"]
    if not isinstance(entries, dict) or set(entries) != set(CORE_TRAJECTORIES):
        raise ValueError("dataset requires exactly four registered complete trajectories")
    for name, entry in entries.items():
        if not isinstance(entry, dict) or set(entry) != {"file", "file_sha256", "metadata", "arrays"}:
            raise ValueError("trajectory manifest does not match the versioned schema")
        _validate_record_identity(name, _strict_json_object(entry["metadata"], "trajectory metadata"))
        for sensor_id, sensor in enumerate(("qpos", "qvel")):
            if entry["metadata"]["noise"][sensor]["seed_sequence_entropy"] != [
                    manifest["seed"], 0, TRAJECTORY_IDS[name], sensor_id]:
                raise ValueError("noise seed keys disagree with the registered dataset identity")
        if entry["file"] != f"{name}.npz":
            raise ValueError("dataset NPZ path must be the registered relative filename")
        if not isinstance(entry["arrays"], dict) or set(entry["arrays"]) != ARRAY_NAMES:
            raise ValueError("trajectory array manifest does not match the versioned schema")
        for descriptor in entry["arrays"].values():
            if (not isinstance(descriptor, dict)
                    or set(descriptor) != {"shape", "dtype", "sha256"}
                    or descriptor["dtype"] != np.dtype(np.float64).str
                    or not isinstance(descriptor["shape"], list)
                    or any(type(dimension) is not int or dimension < 1
                           for dimension in descriptor["shape"])
                    or not _valid_sha256(descriptor["sha256"])):
                raise ValueError("invalid array descriptor")
        if not _valid_sha256(entry["file_sha256"]):
            raise ValueError("invalid file SHA256 descriptor")
    return manifest_path.parent, manifest


def _valid_sha256(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _read_arrays(directory: Path, entry: Mapping[str, Any], *, training_only: bool
                 ) -> Mapping[str, FloatArray]:
    path = directory / entry["file"]
    if path.is_symlink() or not path.is_file():
        raise ValueError("dataset NPZ must be a regular file")
    if sha256_file(path) != entry["file_sha256"]:
        raise ValueError(f"dataset file hash mismatch: {path.name}")
    names = TRAIN_ARRAY_NAMES if training_only else tuple(sorted(ARRAY_NAMES))
    arrays: dict[str, FloatArray] = {}
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != ARRAY_NAMES or len(archive.files) != len(ARRAY_NAMES):
                raise ValueError("NPZ array keys do not match the versioned schema")
            for name in names:
                # NpzFile is lazy: only the selected training arrays are loaded.
                array = archive[name]
                descriptor = entry["arrays"][name]
                if (list(array.shape) != descriptor["shape"]
                        or array.dtype.str != descriptor["dtype"]
                        or array_sha256(array) != descriptor["sha256"]):
                    raise ValueError(f"dataset array hash/shape/dtype mismatch: {name}")
                arrays[name] = _owned_readonly(array)
    except (OSError, EOFError, KeyError) as error:
        raise ValueError("invalid dataset NPZ archive") from error
    result = MappingProxyType(arrays)
    _validate_timed_arrays(result, entry["metadata"], full=not training_only)
    return result


def load_training_sequences(path: Path) -> TrainingDataset:
    """Audit and expose training-only inputs without opening held-out files.

    The complete manifest's split identities are checked before any arrays are
    read.  Full latent/held-out integrity is the evaluation reader's job.  File
    hashes are verified only for the two training archives; selected array
    hashes are verified as each lazy array is loaded with pickle disabled.
    """
    directory, manifest = _read_manifest(path)
    sequences = []
    for name in TRAIN_TRAJECTORIES:
        entry = manifest["trajectories"][name]
        arrays = _read_arrays(directory, entry, training_only=True)
        sequences.append(TrainingSequence(
            trajectory_name=name, trajectory_id=entry["metadata"]["trajectory_id"],
            **arrays,
        ))
    return TrainingDataset(seed=manifest["seed"], sequences=tuple(sequences))


def load_evaluation_dataset(path: Path) -> Dataset:
    """Load all arrays for evaluation and verify every file/array/schema hash."""
    directory, manifest = _read_manifest(path)
    records = {}
    for name in CORE_TRAJECTORIES:
        entry = manifest["trajectories"][name]
        records[name] = TrajectoryRecord(
            _read_arrays(directory, entry, training_only=False),
            MappingProxyType(entry["metadata"]),
        )
    dataset = Dataset(seed=manifest["seed"], scenario=manifest["scenario"],
                      records=MappingProxyType(records),
                      metadata=MappingProxyType(manifest["metadata"]),
                      dataset_schema_version=manifest["dataset_schema_version"])
    validate_dataset(dataset)
    return dataset
