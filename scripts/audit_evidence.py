#!/usr/bin/env python3
"""Audit retained synthetic evidence without executing an optimizer.

The only pickle inputs are this repository's authored official result files,
and their retained SHA256 is verified before deserialization. Physical
predictions are independently recomputed with fresh CPU MjData/mj_forward;
metrics and normalized residuals use direct NumPy formulas.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tarfile
import tempfile
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["MPLBACKEND"] = "Agg"
os.environ["MPLCONFIGDIR"] = str(ROOT / "artifacts/matplotlib")

import mujoco
import numpy as np
from mujoco_sysid_lab import advanced_data
from mujoco_sysid_lab.dataset import TRAIN_ARRAY_NAMES, array_sha256
from mujoco_sysid_lab.experiment_config import config_to_dict, load_ground_truth, load_study_config
from mujoco_sysid_lab.full_evaluation import phase_lag_for_signals
from mujoco_sysid_lab.provenance import collect_provenance, sha256_file, snapshot_sources, write_json
from mujoco_sysid_lab.simulation import compile_model


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    def unique(pairs):
        result = {}
        for name, value in pairs:
            require(name not in result, f"Duplicate JSON field: {name}")
            result[name] = value
        return result

    value = json.loads(Path(path).read_text(), parse_constant=lambda value: (_ for _ in ()).throw(
        ValueError(f"Nonfinite JSON {value}")), object_pairs_hook=unique)
    json.dumps(value, allow_nan=False)
    return value


def safe_path(root, relative):
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), f"Evidence reference escapes root: {relative}")
    return path


def verified_ref(root, reference):
    path = safe_path(root, reference["path"])
    require(path.is_file() and sha256_file(path) == reference["sha256"], f"Reference hash mismatch: {path}")
    return path


def archive_commit(root, commit, destination):
    result = subprocess.run(["git", "-C", str(root), "archive", "--format=tar", commit],
                            capture_output=True, check=True)
    with tarfile.open(fileobj=io.BytesIO(result.stdout)) as archive:
        for member in archive.getmembers():
            if member.isfile():
                path = safe_path(destination, member.name)
                path.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source:
                    path.write_bytes(source.read())
            else:
                require(member.isdir(), "Git source archive must not contain symlinks")


def verify_source(root, provenance):
    """Check original source bytes, independent of the current checkout."""
    required = {"git", "source_sha256", "config", "model", "lock", "command", "python",
                "environment", "package_versions", "mujoco_version"}
    require(required <= provenance.keys(), "Incomplete retained source provenance")
    require(provenance["command"]["argv"], "Missing retained command")
    commit = provenance["git"]["commit"]
    with tempfile.TemporaryDirectory(prefix="sysid_source_audit_") as temporary:
        tree = Path(temporary)
        archive_commit(root, commit, tree)
        if provenance["git"]["dirty"]:
            snapshot = provenance.get("source_snapshot")
            require(snapshot is not None, "Dirty source lacks reconstructable snapshot")
            directory = safe_path(root, snapshot["path"])
            manifest = directory / "manifest.json"
            require(sha256_file(manifest) == snapshot["manifest_sha256"], "Dirty snapshot manifest changed")
            data = read_json(manifest)
            require(data["base_commit"] == commit == snapshot["base_commit"], "Dirty snapshot base commit differs")
            patch = verified_ref(directory, data["tracked_patch"])
            require(sha256_file(patch) == snapshot["tracked_patch_sha256"], "Dirty snapshot patch differs")
            if patch.stat().st_size:
                subprocess.run(["git", "apply", "--binary", "--whitespace=nowarn", str(patch)],
                               cwd=tree, check=True, capture_output=True)
            for relative in data["untracked_sources"]:
                original = safe_path(directory / "untracked", relative)
                target = safe_path(tree, relative)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(original.read_bytes())
            require(data["source_sha256"] == snapshot["source_sha256"] == provenance["source_sha256"],
                    "Dirty snapshot/source hash ledgers differ")
        for relative, expected in provenance["source_sha256"].items():
            path = safe_path(tree, relative)
            require(path.is_file() and sha256_file(path) == expected, f"Original source mismatch: {relative}")
        for name in ("config", "model", "lock"):
            verified_ref(tree, provenance[name])
        require(read_json(tree / provenance["config"]["path"]) == provenance["config"]["content"],
                "Retained exact config differs from original source")
    require(provenance["mujoco_version"] == "3.15.0" and provenance["python"]["version"] == "3.12.5",
            "Evidence violates pinned simulator/interpreter")
    require(provenance["environment"]["cpu"] and provenance["environment"]["os"], "Missing CPU/OS")
    return {"commit": commit, "dirty": provenance["git"]["dirty"],
            "source_files_verified": len(provenance["source_sha256"]),
            "config_sha256": provenance["config"]["sha256"], "model_sha256": provenance["model"]["sha256"],
            "lock_sha256": provenance["lock"]["sha256"], "command": provenance["command"],
            "method": "original Git archive plus retained binary patch and authored untracked snapshot"}


def direct_rollout(model, times, controls):
    """Fresh CPU data; state observations at t[k], no fitting/extra force."""
    require((model.nq, model.nv, model.nu, model.nsensordata) == (1, 1, 1, 2), "Model layout differs")
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.ctrl[:] = controls[0]
    mujoco.mj_forward(model, data)
    observations = np.empty((len(times), 2))
    for index in range(len(times)):
        observations[index] = data.sensordata
        require(np.array_equal(data.sensordata, [data.qpos[0], data.qvel[0]]), "Fresh sensor/state mismatch")
        if index + 1 < len(times):
            data.ctrl[:] = controls[index]
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
    require(np.isfinite(observations).all(), "Nonfinite independent rollout")
    return observations


def predicted_observations(model, arrays, delay, prehistory_s):
    times = arrays["timestamps"]
    physical = direct_rollout(model, times, arrays["controls"])
    history_times = np.arange(round(prehistory_s / model.opt.timestep) + 1) * model.opt.timestep
    history = direct_rollout(model, history_times, np.zeros((len(history_times), 1)))
    require(np.array_equal(history, np.zeros_like(history)), "Candidate prehistory is not equilibrium")
    support = np.concatenate((history_times[:-1] - prehistory_s, times))
    values = np.vstack((history[:-1], physical))
    query = times - delay
    require(query[0] >= support[0] and query[-1] <= support[-1], "Audit delay would extrapolate")
    observed = np.column_stack([np.interp(query, support, values[:, channel]) for channel in range(2)])
    return observed


def manual_metrics(arrays, prediction, study):
    times = arrays["timestamps"]
    tolerance = 1e-12 + 8 * np.finfo(float).eps * len(times) * max(1., float(times[-1]))
    window = (.06, float(times[-1]) - .06)
    mask = (times >= window[0] - tolerance) & (times <= window[1] + tolerance)
    measured = np.column_stack((arrays["measured_qpos"][:, 0], arrays["measured_qvel"][:, 0]))
    errors = measured[mask] - prediction[mask]
    residual = errors / np.array([study.sensor_sigmas["qpos"], study.sensor_sigmas["qvel"]])
    metrics = {"qpos_rmse": float(np.sqrt(np.mean(errors[:, 0] ** 2))),
               "qvel_rmse": float(np.sqrt(np.mean(errors[:, 1] ** 2))),
               "normalized_rmse": float(np.sqrt(np.mean(residual ** 2))),
               "qpos_p95_absolute_error": float(np.percentile(np.abs(errors[:, 0]), 95, method="linear")),
               "qvel_p95_absolute_error": float(np.percentile(np.abs(errors[:, 1]), 95, method="linear")),
               "n_samples": int(mask.sum()), "objective": float(.5 * np.sum(residual ** 2))}
    return metrics, residual.ravel()


METRIC_KEYS = ("qpos_rmse", "qvel_rmse", "normalized_rmse", "qpos_p95_absolute_error", "qvel_p95_absolute_error")


def close(actual, expected, label, *, atol=1e-10, rtol=1e-10):
    require(np.allclose(actual, expected, atol=atol, rtol=rtol), f"Numerical disagreement: {label}")


def audit_metrics(data, model, delay, study, retained=None):
    metrics, residuals, predictions = {}, [], {}
    for name, record in data.records.items():
        prediction = predicted_observations(model, record.arrays, delay, study.prehistory_s)
        predictions[name] = prediction
        metrics[name], residual = manual_metrics(record.arrays, prediction, study)
        if name in data.training_ids:
            residuals.append(residual)
    if retained is not None:
        compare_metric_rows(data, metrics, retained)
    return metrics, np.concatenate(residuals), predictions


def compare_metric_rows(data, metrics, retained):
    for split in ("train", "heldout"):
        names = data.training_ids if split == "train" else tuple(data.heldout)
        for key in METRIC_KEYS[:3]:
            aggregate = np.sqrt(np.mean([metrics[name][key] ** 2 for name in names]))
            close(aggregate, retained[split][key], f"{split} aggregate {key}")
        rows = retained[split]["trajectory_metrics"]
        require(len(rows) == len(names), "Metric split omits or duplicates complete trajectories")
        for row in rows:
            name = row.get("trajectory_name", row.get("trajectory", row.get("trajectory_id")))
            require(name in names, "Retained metric split contains wrong trajectory")
            for key in (*METRIC_KEYS, "n_samples"):
                close(metrics[name][key], row[key], f"{name} {key}")


def audit_phase(data, predictions, study, phase):
    for name, channels in phase["trajectory_diagnostics"].items():
        arrays = data.records[name].arrays
        for sensor_id, sensor in enumerate(("qpos", "qvel")):
            audited = phase_lag_for_signals(arrays["timestamps"], predictions[name][:, sensor_id:sensor_id + 1],
                    arrays["measured_" + sensor], study.sensor_sigmas[sensor], study)
            saved = channels["channels"][sensor]
            require(audited["signed_lag_s"] == saved["signed_lag_s"] and audited["valid"] == saved["valid"], "Phase lag/sign differs")


def source_fit(root, retained):
    path = verified_ref(root, retained["source_result"])
    raw = read_json(path)
    if path.name == "run.json":
        fit_id = retained.get("source_fit_id", retained["fit_id"])
        row = next(row for row in raw["fits"] if row["fit_id"] == fit_id)
        return row, path.parent / "fits" / fit_id
    return raw, path.parent


def fit_ref(root, directory, reference):
    relative = reference["path"]
    base = directory.parents[1] if relative.startswith("fits/") else directory
    return verified_ref(base, reference)


def audit_model(model, physical):
    close(model.body_mass[1], physical["mass_kg"], "MJB fitted mass", atol=1e-14, rtol=1e-14)
    close(model.dof_damping[0], physical["damping_nm_s_per_rad"], "MJB fitted damping", atol=1e-14, rtol=1e-14)
    mass, length, radius = physical["mass_kg"], physical["link_length_m"], physical["link_radius_m"]
    close(model.body_inertia[1], [mass * (length ** 2 / 12 + radius ** 2 / 4)] * 2 + [mass * radius ** 2 / 2],
          "consistent COM inertia", atol=1e-14, rtol=1e-14)
    close(model.body_ipos[1], [0., 0., -length / 2], "fixed COM")
    close(model.actuator_gear[0], [physical["motor_gain_nm_per_unit"], 0, 0, 0, 0, 0], "known gain")
    require(np.array_equal(model.dof_armature, [0.]) and np.array_equal(model.dof_frictionloss, [0.]),
            "Fitting model contains armature/dry friction")
    require(np.array_equal(model.dof_dampingpoly, [[0., 0.]]), "Nonlinear damping entered fitting family")
    close(model.opt.timestep, physical["timestep_s"], "timestep")
    require(model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST, "Wrong integrator")
    close(model.opt.gravity, [0., 0., -9.81], "fixed gravity")


def audit_datasets(root, run, study, truth):
    datasets, array_count, training_loads = {}, 0, []
    for key, reference in run["datasets"].items():
        manifest = verified_ref(root, reference)
        raw = read_json(manifest)
        require(raw["provenance"]["config"]["sha256"] == run["provenance"]["config"]["sha256"], "Dataset config differs")
        require(raw["metadata"]["truth_config_sha256"] == sha256_file(study.truth_config_path), "Dataset truth config differs")
        require(raw["metadata"]["model_source_sha256"] == run["provenance"]["model"]["sha256"], "Dataset MJCF differs")
        calls = []
        original = advanced_data._read_arrays

        def traced(directory, entry, *, training_only):
            arrays = original(directory, entry, training_only=training_only)
            calls.append({"file": entry["file"], "training_only": training_only, "arrays": sorted(arrays)})
            return arrays

        try:
            advanced_data._read_arrays = traced
            training = advanced_data.load_scenario_training(manifest)
        finally:
            advanced_data._read_arrays = original
        names = ("weak_step",) if raw["scenario"] == "weak" else ("train_multisine", "train_prbs")
        require(len(calls) == len(names) and {call["file"] for call in calls} ==
                {raw["trajectories"][name]["file"] for name in names}, "Training loader opened a held-out file")
        require(all(call["training_only"] and set(call["arrays"]) == set(TRAIN_ARRAY_NAMES) for call in calls),
                "Training loader exposed hidden arrays")
        require(tuple(sequence.trajectory_name for sequence in training.sequences) == names, "Training whitelist differs")
        training_loads.append({"dataset": key, "files_opened": calls})
        data = advanced_data.load_scenario(manifest)
        for name, record in data.records.items():
            arrays = record.arrays
            array_count += len(arrays)
            require(np.array_equal(arrays["initial_state"], [0., 0., 0.]), "Initial state differs")
            for sensor_id, sensor in enumerate(("qpos", "qvel")):
                noise = record.metadata["noise"][sensor]
                entropy = [data.seed, 1 if name == "weak_step" else 0, study.trajectories[name].trajectory_id, sensor_id]
                require(tuple(noise["seed_sequence_entropy"]) == tuple(entropy), "Noise seed derivation differs")
                expected = np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy))).standard_normal(arrays["innovation_" + sensor].shape)
                require(np.array_equal(expected, arrays["innovation_" + sensor]), "Noise innovations differ from registered PRNG")
                require(np.array_equal(arrays["measured_" + sensor], arrays["observed_clean_" + sensor]
                                       + study.sensor_sigmas[sensor] * expected), "Noise was not added after clean delay")
                require(np.array_equal(arrays["prehistory_" + sensor], np.zeros_like(arrays["prehistory_" + sensor])), "Nonzero equilibrium history")
                support = np.concatenate((arrays["prehistory_timestamps"][:-1], arrays["timestamps"]))
                signal = np.concatenate((arrays["prehistory_" + sensor][:-1, 0], arrays["clean_" + sensor][:, 0]))
                delayed = np.interp(arrays["timestamps"] - data.delay_s, support, signal)
                require(np.array_equal(delayed, arrays["observed_clean_" + sensor][:, 0]), "Stored clean-delay sign/support differs")
            if data.scenario == "mismatch":
                mismatch = truth.mismatch
                expected = -mismatch["torque_nm"] * np.tanh(arrays["clean_qvel"] / mismatch["velocity_scale_rad_s"])
                require(np.array_equal(expected, arrays["extra_torque_nm"]), "Omitted-force formula differs")
            else:
                require(np.array_equal(arrays["extra_torque_nm"], np.zeros_like(arrays["extra_torque_nm"])), "Omitted force leaked into matched family")
        datasets[key] = data
    for key, data in datasets.items():
        matched = datasets.get(f"matched_seed{data.seed}")
        if matched is not None:
            for name in set(data.records) & set(matched.records):
                for sensor in ("qpos", "qvel"):
                    require(np.array_equal(data.records[name].arrays["innovation_" + sensor],
                                           matched.records[name].arrays["innovation_" + sensor]), "Paired noise differs across scenarios")
    return datasets, {"dataset_count": len(datasets), "arrays_verified": array_count,
                      "training_loader_isolation": training_loads}


def audit_run(root, run_path):
    started = perf_counter()
    run_path = Path(run_path)
    if run_path.is_dir():
        run_path /= "run.json"
    run = read_json(run_path)
    require(run["schema_version"] == "mujoco-sysid-lab.full-run.v1", "Unsupported evidence run schema")
    require(run["status"] == "passed", "Evidence execution is incomplete; audit cannot certify completeness")
    source_checks, seen = [], set()

    def provenance_check(value):
        key = (value["git"]["commit"], json.dumps(value["source_sha256"], sort_keys=True))
        if key not in seen:
            source_checks.append(verify_source(root, value))
            seen.add(key)

    provenance_check(run["provenance"])
    for name, version in run["provenance"]["package_versions"].items():
        require(importlib.metadata.version(name) == version, f"Audit library differs: {name}")
    study = load_study_config(root / run["provenance"]["config"]["path"])
    truth = load_ground_truth(study.truth_config_path, study)
    require(sha256_file(root / run["provenance"]["model"]["path"]) == run["provenance"]["model"]["sha256"],
            "Current model differs; audit must use original source")
    require(sha256_file(root / run["provenance"]["config"]["path"]) == run["provenance"]["config"]["sha256"],
            "Current config differs; audit must use original source")
    datasets, dataset_checks = audit_datasets(root, run, study, truth)
    specifications = {row["id"]: row for row in study.evidence_matrix}
    cells = ("B", "delay_excluded", "C_delay_only", "C_joint") if run["through_milestone"] == 6 else (
        "A1", "A2", "B", "weak", "delay_excluded", "C_delay_only", "C_joint", *(('E',) if run["through_milestone"] >= 8 else ()))
    expected = {f"{cell}_seed{seed}_{start}" for cell in cells
                for seed in ((101,) if run["through_milestone"] == 6 else specifications[cell]["noise_seeds"])
                for start in specifications[cell]["start_ids"]}
    require(len(run["fits"]) == len(expected) and {row["fit_id"] for row in run["fits"]} == expected,
            "Registered matrix missing, duplicated, or unexpected fits")
    require(run["optimization_calls_this_invocation"] + run["reused_registered_fits"] == len(expected), "Invocation/reuse accounting differs")
    fitted_rows, predictions = {}, {}
    official_file_count = 0
    for retained in run["fits"]:
        fit_id, cell = retained["fit_id"], retained["cell"]
        raw, directory = source_fit(root, retained)
        for key in ("provenance", "optimization_provenance"):
            if key in raw:
                provenance_check(raw[key])
        free = tuple(retained["free_parameters"])
        require(free == tuple(specifications[cell]["free_parameters"]), f"Wrong parameter order: {fit_id}")
        require(retained["start_id"] in specifications[cell]["start_ids"], f"Wrong start identity: {fit_id}")
        scenario = "weak" if cell == "weak" else specifications[cell]["scenario"]
        data = datasets[f"{scenario}_seed{retained['seed']}"]
        require(raw["input_array_sha256"] == {sequence.trajectory_name:
                {name: array_sha256(getattr(sequence, name)) for name in TRAIN_ARRAY_NAMES}
                for sequence in data.training.sequences}, f"Estimator input differs: {fit_id}")
        official = directory / "official"
        files = raw["official_outputs"]["files"]
        for reference in files.values():
            verified_ref(official, reference)
            official_file_count += 1
        native_path = verified_ref(official, files["results.pkl"])
        native = pickle.loads(native_path.read_bytes())
        if "native_result" in raw:
            original_native = pickle.loads(fit_ref(root, directory, raw["native_result"]).read_bytes())
            require(np.array_equal(native.x, original_native.x) and np.array_equal(native.fun, original_native.fun)
                    and np.array_equal(native.jac, original_native.jac), f"Official/native results differ: {fit_id}")
        residual_path = fit_ref(root, directory, raw["residuals"]) if "residuals" in raw else directory / "residuals.npz"
        if "residuals_sha256" in raw:
            require(sha256_file(residual_path) == raw["residuals_sha256"], f"Residual hash changed: {fit_id}")
        with np.load(residual_path, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        require(all(np.isfinite(value).all() for value in arrays.values()), f"Nonfinite raw fit: {fit_id}")
        require(np.array_equal(arrays["final"], native.fun) and np.array_equal(arrays["jacobian"], native.jac)
                and np.array_equal(arrays["optimized_vector"], native.x), f"NPZ/native arrays differ: {fit_id}")
        close(.5 * arrays["final"] @ arrays["final"], native.cost, f"Native objective {fit_id}")
        close(native.x, retained["optimized_vector"], f"Retained vector {fit_id}", atol=0., rtol=0.)
        close(native.cost, retained["final_objective"], f"Retained objective {fit_id}")
        model = mujoco.MjModel.from_binary_path(str(verified_ref(official, files["pendulum.mjb"])))
        physical = retained.get("fitted_physics", raw.get("fitted_physics"))
        audit_model(model, physical)
        for name, value in config_to_dict(study.nominal).items():
            if name not in ("mass_kg", "damping_nm_s_per_rad"):
                require(physical[name] == value, f"Fixed nuisance changed: {fit_id}/{name}")
        delay = retained.get("fitted_delay_s", 0.)
        _, residual, prediction = audit_metrics(data, model, delay, study, retained.get("metrics"))
        close(residual, arrays["final"], f"Independent final residual {fit_id}", atol=1e-8, rtol=1e-10)
        summary = retained.get("optimization_summary", raw.get("summary"))
        require(native.nfev == summary["optimizer"]["nfev"] and native.njev == summary["optimizer"]["njev"],
                f"Native evaluation counts differ: {fit_id}")
        counts = summary["call_counts"]
        require(all(counts["total"][name] == sum(counts[phase][name] for phase in ("initial", "optimization", "final", "external"))
                    for name in counts["total"]), f"Residual ledger totals differ: {fit_id}")
        require(counts["optimization"]["logical_rollouts"] == counts["optimization"]["candidate_columns"] * len(data.training_ids),
                f"Logical rollout count differs: {fit_id}")
        require(native.jac.shape == (len(residual), len(free)), f"Jacobian shape differs: {fit_id}")
        sensitivity = summary["sensitivity"]
        singular = np.linalg.svd(native.jac * np.array(sensitivity["column_scale_values"]), compute_uv=False)
        close(singular, sensitivity["singular_values"], f"Scaled sensitivity singular values {fit_id}")
        require(int(np.count_nonzero(singular > sensitivity["rank_tolerance"])) == sensitivity["rank"],
                f"Sensitivity rank differs: {fit_id}")
        for parameter in retained["parameter_rows"]:
            value = dict(zip(free, native.x, strict=True))[parameter["name"]]
            expected_truth = data.delay_s if parameter["name"] == "delay_s" else getattr(truth.model, parameter["name"])
            close(parameter["truth"], expected_truth, f"Parameter truth {fit_id}")
            close(parameter["absolute_error"], abs(value - expected_truth), f"Parameter absolute error {fit_id}")
            width = parameter["upper_bound"] - parameter["lower_bound"]
            hit = (value - parameter["lower_bound"] <= .001 * width or parameter["upper_bound"] - value <= .001 * width)
            require(hit == parameter["bound_hit"], f"Bound-width hit rule differs: {fit_id}")
        fitted_rows[fit_id] = {"fit_id": fit_id, "cell": cell, "seed": retained["seed"],
                "start_id": retained["start_id"], "final_objective": float(native.cost),
                "optimized_vector": native.x.tolist(), "source_result": retained["source_result"],
                "native_nfev": int(native.nfev), "native_njev": int(native.njev),
                "jacobian_shape": list(native.jac.shape), "optimization_call_counts": counts["optimization"],
                "maximum_independent_residual_error": float(np.max(np.abs(residual - arrays["final"]))),
                "optimizer_success": bool(native.success), "train_only_inputs_verified": True}
        predictions[fit_id] = prediction
    row_index = {row["fit_id"]: row for row in run["fits"]}
    group_checks = []
    for group in run["groups"]:
        seal = read_json(verified_ref(root, group["selection"]))
        rows = [row_index[name] for name in group["fits"]]
        winner = min(rows, key=lambda row: (row["final_objective"], ("S0", "S1", "S2").index(row["start_id"])))
        require(seal["selected_fit_id"] == group["selected_fit_id"] == winner["fit_id"], "Selection did not minimize training cost")
        require(seal["selected_vector"] == winner["optimized_vector"] and seal["selected_final_objective"] == winner["final_objective"],
                "Selection vector/cost differs from immutable fit")
        require(seal["physical_truth_evaluated"] is False and seal["heldout_evaluated"] is False, "Selection used evaluation data")
        data = datasets[f"{group['scenario']}_seed{group['seed']}"]
        for label, physical, delay in (("nominal", study.nominal, 0.), ("oracle", truth.model, data.delay_s)):
            audit_metrics(data, compile_model(root / run["provenance"]["model"]["path"], physical), delay, study, group[label])
        audit_phase(data, predictions[winner["fit_id"]], study, group["phase_lag"])
        noise_metrics = {name: manual_metrics(record.arrays, np.column_stack((record.arrays["observed_clean_qpos"][:, 0],
                record.arrays["observed_clean_qvel"][:, 0])), study)[0] for name, record in data.records.items()}
        compare_metric_rows(data, noise_metrics, group["generator_aware_noise_floor"])
        if group["cell"] == "C_delay_only":
            bgroup = next(row for row in run["groups"] if row["group_id"] == f"B_seed{group['seed']}")
            bfit = row_index[bgroup["selected_fit_id"]]
            require(group["fixed_physics_source_fit_id"] == bfit["fit_id"], "Delay fixed physics is not same-seed B winner")
            require(all(abs(bfit["fitted_physics"][name] / getattr(truth.model, name) - 1) <= .1 for name in ("mass_kg", "damping_nm_s_per_rad")),
                    "Delay-only bypassed physical-recovery prerequisite")
            require(all(winner["fitted_physics"][name] == bfit["fitted_physics"][name] for name in ("mass_kg", "damping_nm_s_per_rad")),
                    "Delay-only changed frozen B physics")
            fixed = replace(study.nominal, **{name: bfit["fitted_physics"][name] for name in ("mass_kg", "damping_nm_s_per_rad")})
            _, _, before = audit_metrics(data, compile_model(root / run["provenance"]["model"]["path"], fixed), 0., study,
                                         group["fixed_physics_zero_delay_metrics"])
            audit_phase(data, before, study, group["fixed_physics_zero_delay_phase_lag"])
        if group["cell"] == "C_joint":
            prior = next(row for row in run["groups"] if row["group_id"] == f"C_delay_only_seed{group['seed']}")
            require(prior["joint_delay_eligible"] and abs(row_index[prior["selected_fit_id"]]["fitted_delay_s"] - truth.delay_study_s) <= .002,
                    "Joint-delay bypassed one-timestep prerequisite")
        group_checks.append({"group_id": group["group_id"], "training_selected_fit_id": winner["fit_id"],
                             "selection_sha256": group["selection"]["sha256"], "baseline_metrics_and_phase_verified": True})
    criteria = {}
    for cell in ("B", "weak", "C_joint", "E"):
        selected = [row_index[group["selected_fit_id"]] for group in run["groups"] if group["cell"] == cell]
        if selected:
            criteria[cell] = {"selected_seed_count": len(selected), "seeds": [row["seed"] for row in selected],
                "physical_relative_errors": {name: [abs(row["fitted_physics"][name] / getattr(truth.model, name) - 1)
                    for row in selected] for name in ("mass_kg", "damping_nm_s_per_rad")},
                "heldout_improvements": {name: [row["improvement_relative_to_nominal"]["heldout"][name] for row in selected]
                    for name in METRIC_KEYS[:3]}}
            if cell == "C_joint":
                criteria[cell]["absolute_delay_errors_s"] = [abs(row["fitted_delay_s"] - truth.delay_study_s) for row in selected]
    return {"schema_version": "mujoco-sysid-lab.evidence-audit.v1", "verdict": "passed",
            "inspected_run": {"path": run_path.relative_to(root).as_posix(), "sha256": sha256_file(run_path)},
            "original_run_provenance": run["provenance"], "audit_method": "No optimizer calls. Original source reconstruction, strict JSON/NPZ/hash verification, fresh full-precision MJB CPU predictions, manual NumPy metrics/residuals, sealed training selection and same-seed prerequisites.",
            "scientific_claim_boundary": "Synthetic workflow evidence only; no real robot, HIL, deployment, or arbitrary-robot identifiability.",
            "source_checks": source_checks, "dataset_checks": dataset_checks,
            "registered_fit_count": len(expected), "official_files_verified": official_file_count,
            "candidate_checks": list(fitted_rows.values()), "group_checks": group_checks,
            "descriptive_criteria_inputs": criteria, "optimization_calls": 0,
            "one_column_condition_does_not_establish_joint_identifiability": True,
            "wall_seconds": perf_counter() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Existing retained run directory or run.json")
    parser.add_argument("--output", required=True, help="New ignored artifacts/audits/*.json")
    args = parser.parse_args()
    run = safe_path(ROOT, args.run)
    output = safe_path(ROOT, args.output)
    require(output.is_relative_to(ROOT / "artifacts/audits") and not output.exists(), "Choose new ignored audit JSON")
    provenance = collect_provenance(ROOT, ROOT / "configs/evidence_v1.json", ROOT / "model/pendulum.xml",
                                    [".venv/bin/python", "scripts/audit_evidence.py", *sys.argv[1:]])
    if provenance["git"]["dirty"]:
        provenance["source_snapshot"] = snapshot_sources(ROOT, output.parent / f"{output.stem}_source_snapshot")
    audit = audit_run(ROOT, run)
    audit["provenance"] = provenance
    write_json(output, audit)
    print(json.dumps({"verdict": audit["verdict"], "fits": audit["registered_fit_count"],
                      "arrays": audit["dataset_checks"]["arrays_verified"], "output": args.output,
                      "wall_seconds": audit["wall_seconds"]}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
