#!/usr/bin/env python3
"""Run the authorized Milestone 3 oracle-assisted single-parameter checks."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
import traceback

ROOT = Path(__file__).resolve().parents[1]
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["MPLBACKEND"] = "Agg"
os.environ["MPLCONFIGDIR"] = str(ROOT / "artifacts/matplotlib")
os.environ["XDG_CACHE_HOME"] = str(ROOT / "artifacts/cache")


def validate_registration(config_path: Path, truth_path: Path, registration_path: Path) -> dict:
    """Reject edited settings before the first identification evidence call."""
    from mujoco_sysid_lab.provenance import sha256_file
    registration = json.loads(registration_path.read_text())
    if registration.get("schema_version") != "mujoco-sysid-lab.registration.v1":
        raise ValueError("Unsupported preregistration schema")
    for name, path in (("evidence", config_path), ("ground_truth", truth_path)):
        if (sha256_file(path) != registration[f"{name}_config_sha256"]
                or json.loads(path.read_text()) != registration[f"{name}_config"]):
            raise ValueError(f"Frozen {name} configuration changed; create and explain a new registered version")
    return registration


def make_noiseless_training(training, truth_model, model_path: Path):
    """Generate only the two training observations for labeled unit verification."""
    from mujoco_sysid_lab.dataset import TrainingDataset, TrainingSequence
    from mujoco_sysid_lab.simulation import compile_model, simulate
    model = compile_model(model_path, truth_model)
    sequences = []
    for sequence in training.sequences:
        simulated = simulate(model, sequence.timestamps, sequence.controls,
                             initial_qpos=float(sequence.initial_state[1]),
                             initial_qvel=float(sequence.initial_state[2]))
        sequences.append(TrainingSequence(
            trajectory_name=sequence.trajectory_name, trajectory_id=sequence.trajectory_id,
            timestamps=sequence.timestamps, controls=sequence.controls,
            measured_qpos=simulated.qpos, measured_qvel=simulated.qvel,
            initial_state=sequence.initial_state,
        ))
    return TrainingDataset(seed=training.seed, sequences=tuple(sequences))


def training_metrics(training, physical, model_path: Path, study) -> dict:
    """Evaluate only training trajectories with the established common evaluator."""
    from mujoco_sysid_lab.evaluation import aggregate_metrics, evaluate_trajectory
    from mujoco_sysid_lab.simulation import compile_model, simulate
    model = compile_model(model_path, physical)
    rows = []
    for sequence in training.sequences:
        predicted = simulate(model, sequence.timestamps, sequence.controls,
                             initial_qpos=float(sequence.initial_state[1]),
                             initial_qvel=float(sequence.initial_state[2]))
        rows.append({"trajectory": sequence.trajectory_name, "split": "train",
                     **evaluate_trajectory(
                         sequence.timestamps, sequence.measured_qpos, sequence.measured_qvel,
                         predicted.qpos, predicted.qvel,
                         study.sensor_sigmas["qpos"], study.sensor_sigmas["qvel"],
                         window_s=study.evaluation_window_for(sequence.trajectory_name))})
    return aggregate_metrics(rows)


def main() -> int:
    started = perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/evidence_v1.json")
    parser.add_argument("--registration", default="docs/results/registration_v1.json")
    parser.add_argument("--milestone", type=int, choices=[3], default=3)
    parser.add_argument("--dataset", help="Existing matched seed-101 dataset manifest or directory")
    parser.add_argument("--output", help="New ignored directory; omit for a UTC-timestamped run")
    args = parser.parse_args()
    config_path = (ROOT / args.config).resolve()
    registration_path = (ROOT / args.registration).resolve()
    model_path = ROOT / "model/pendulum.xml"
    output = (ROOT / (args.output or ("artifacts/milestone3/" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")))).resolve()
    if not output.is_relative_to(ROOT / "artifacts"):
        parser.error("Output must be under ignored artifacts/")
    if output.exists():
        parser.error("Output already exists. Omit --output for a fresh run; existing evidence is never overwritten.")

    import numpy as np
    from mujoco_sysid_lab.dataset import array_sha256, load_training_sequences
    from mujoco_sysid_lab.experiment_config import load_ground_truth, load_study_config
    from mujoco_sysid_lab.fit_reporting import save_official_fit, summarize_fit
    from mujoco_sysid_lab.identification import build_problem
    from mujoco_sysid_lab.provenance import collect_provenance, sha256_file, snapshot_sources, write_json

    try:
        study = load_study_config(config_path)
        registration = validate_registration(config_path, study.truth_config_path, registration_path)
        truth = load_ground_truth(study.truth_config_path, study)
        executable = Path(sys.executable).absolute()
        executable_text = executable.relative_to(ROOT).as_posix() if executable.is_relative_to(ROOT) else str(executable)
        provenance = collect_provenance(ROOT, config_path, model_path,
                                        [executable_text, "scripts/run_experiment.py", *sys.argv[1:]])
        if provenance["mujoco_version"] != "3.15.0" or provenance["python"]["version"] != "3.12.5":
            raise ValueError("Restore registered MuJoCo 3.15.0 / Python 3.12.5 environment")
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    output.mkdir(parents=True, exist_ok=False)
    provenance.update(seed=101, milestone=3, scenario="matched",
                      created_at_utc=datetime.now(timezone.utc).isoformat(),
                      truth_config={"path": study.truth_config_path.relative_to(ROOT).as_posix(),
                                    "sha256": sha256_file(study.truth_config_path),
                                    "content": json.loads(study.truth_config_path.read_text())},
                      preregistration={"path": registration_path.relative_to(ROOT).as_posix(),
                                       "sha256": sha256_file(registration_path)})
    if provenance["git"]["dirty"]:
        provenance["source_snapshot"] = snapshot_sources(ROOT, output / "source_snapshot")
    record = {"schema_version": "mujoco-sysid-lab.milestone3.v1", "status": "running",
              "provenance": provenance, "fits": [], "identification_performed": False,
              "scope": "Oracle-assisted independent mass and damping checks; no joint recovery or held-out scoring."}
    write_json(output / "run.json", record)
    write_json(output / "registration.json", registration)
    try:
        dataset = Path(args.dataset).resolve() if args.dataset and Path(args.dataset).is_absolute() else (ROOT / args.dataset).resolve() if args.dataset else ROOT / "artifacts/milestone2/20261012_seed101_verified/dataset"
        if args.dataset is None and not dataset.exists():
            # A fresh clone still has a one-command CPU path. This child run is
            # dataset generation, and no held-out arrays enter the estimator.
            dataset_run = output / "dataset_input"
            child = [executable_text, "scripts/generate_data.py", "--config", args.config,
                     "--output", dataset_run.relative_to(ROOT).as_posix()]
            with (output / "dataset_generation.log").open("w") as log:
                subprocess.run(child, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            dataset = dataset_run / "dataset"
        manifest_path = dataset / "manifest.json" if dataset.is_dir() else dataset
        manifest = json.loads(manifest_path.read_text())
        if (not manifest_path.resolve().is_relative_to(ROOT / "artifacts")
                or manifest["seed"] != 101 or manifest["scenario"] != "matched"
                or manifest["provenance"]["config"]["sha256"] != sha256_file(config_path)
                or manifest["provenance"]["model"]["sha256"] != sha256_file(model_path)
                or manifest["metadata"]["truth_config_sha256"] != sha256_file(study.truth_config_path)):
            raise ValueError("Dataset must be authored matched seed-101 data with the frozen config, truth, and model")
        training = load_training_sequences(manifest_path)
        for sequence in training.sequences:
            expected_samples = int(round(study.trajectories[sequence.trajectory_name].duration_s / study.timestep_s)) + 1
            if len(sequence.timestamps) != expected_samples:
                raise ValueError("Registered evidence requires the complete eight-second training horizons")
        record["dataset_input"] = {"path": manifest_path.relative_to(ROOT).as_posix(),
                                   "sha256": sha256_file(manifest_path),
                                   "generation_provenance": manifest["provenance"],
                                   "training_ids": [sequence.trajectory_name for sequence in training.sequences],
                                   "heldout_arrays_loaded": False}
        noiseless = make_noiseless_training(training, truth.model, model_path)
        truth_values = {"mass_kg": truth.model.mass_kg,
                        "damping_nm_s_per_rad": truth.model.damping_nm_s_per_rad}
        start = {"mass_kg": study.starts["S0"].mass_kg,
                 "damping_nm_s_per_rad": study.starts["S0"].damping_nm_s_per_rad}
        sanity_passed = True
        for clean_check, inputs in ((True, noiseless), (False, training)):
            for cell, parameter in (("A1", "mass_kg"), ("A2", "damping_nm_s_per_rad")):
                fit_id = f"{cell}_{'noiseless_sanity' if clean_check else 'seed101_S0'}"
                row = {"schema_version": "mujoco-sysid-lab.fit.v1", "fit_id": fit_id,
                       "cell": cell, "status": "running", "provenance": provenance,
                       "registered_evidence_fit": not clean_check, "noiseless_sanity": clean_check,
                       "seed": 101, "start_id": "S0", "free_parameters": [parameter],
                       "oracle_assisted": True, "truth_visibility": "Fixed nuisance physics stamped by this unit-check harness; free parameter starts at nominal.",
                       "heldout_evaluated": False}
                if not clean_check and not sanity_passed:
                    row.update(status="not_run", reason="Noiseless correctness prerequisite failed")
                    record["fits"].append(row)
                    write_json(output / "run.json", record)
                    continue
                fit_dir = output / "fits" / fit_id
                fit_dir.mkdir(parents=True, exist_ok=False)
                write_json(fit_dir / "result.json", row)
                problem = None
                execution = None
                try:
                    fixed = (replace(study.nominal, damping_nm_s_per_rad=truth.model.damping_nm_s_per_rad)
                             if cell == "A1" else replace(study.nominal, mass_kg=truth.model.mass_kg))
                    row["initial_physics"] = asdict(fixed)
                    row["optimizer_options"] = dict(study.optimizer)
                    problem = build_problem(inputs, model_path, fixed, study, (parameter,), {parameter: start[parameter]})
                    # This reference evaluation checks the observation plumbing;
                    # the reference vector is never used as an optimizer start.
                    reference = problem.evaluate(np.array([truth_values[parameter]]))
                    expected = []
                    for observed, clean in zip(inputs.sequences, noiseless.sequences, strict=True):
                        lo, hi = study.evaluation_window_for(observed.trajectory_name)
                        mask = (observed.timestamps >= lo - 1e-12) & (observed.timestamps <= hi + 1e-12)
                        expected.append(np.column_stack((
                            (observed.measured_qpos[mask, 0] - clean.measured_qpos[mask, 0]) / study.sensor_sigmas["qpos"],
                            (observed.measured_qvel[mask, 0] - clean.measured_qvel[mask, 0]) / study.sensor_sigmas["qvel"])).ravel())
                    expected_residual = np.concatenate(expected)
                    error = float(np.max(np.abs(reference - expected_residual)))
                    if error > 1e-10:
                        raise RuntimeError(f"Noise-weighted reference residual mismatch: {error}")
                    row["truth_reference_residual_max_abs_error"] = error
                    row["input_array_sha256"] = {}
                    for sequence in inputs.sequences:
                        arrays = {name: getattr(sequence, name) for name in (
                            "timestamps", "controls", "measured_qpos", "measured_qvel", "initial_state")}
                        np.savez_compressed(fit_dir / f"{sequence.trajectory_name}_inputs.npz", **arrays)
                        row["input_array_sha256"][sequence.trajectory_name] = {name: array_sha256(value) for name, value in arrays.items()}
                    row["optimization_attempted"] = True
                    write_json(fit_dir / "result.json", row)
                    with (fit_dir / "optimizer.log").open("w") as log, redirect_stdout(log):
                        execution = problem.run()
                    row["optimization_invoked"] = problem.optimization_invoked
                    record["identification_performed"] |= problem.optimization_invoked
                    summary = summarize_fit(problem, execution, truth_values,
                                            noiseless=clean_check,
                                            bound_hit_fraction=study.metrics["bound_hit_fraction"])
                    row["summary"] = summary
                    fitted = replace(fixed, **{parameter: float(execution.optimized_params[parameter].value[0])})
                    row["fitted_physics"] = asdict(fitted)
                    row["training_metrics"] = {
                        "initial_unit_model": training_metrics(inputs, fixed, model_path, study),
                        "identified_unit_model": training_metrics(inputs, fitted, model_path, study),
                        "oracle": training_metrics(inputs, truth.model, model_path, study),
                    }
                    identified_metrics = row["training_metrics"]["identified_unit_model"]
                    metric_objective = sum(item["n_samples"] for item in identified_metrics["trajectory_metrics"]) * identified_metrics["normalized_rmse"] ** 2
                    official_objective = float(0.5 * np.dot(execution.final_residual, execution.final_residual))
                    if not np.isclose(metric_objective, official_objective, rtol=1e-9, atol=1e-8):
                        raise RuntimeError("Shared metric objective disagrees with the official optimizer residual")
                    row["metric_objective_consistency"] = {"shared_metric_objective": metric_objective,
                                                            "official_residual_objective": official_objective,
                                                            "passed": True}
                    np.savez_compressed(fit_dir / "residuals.npz",
                                        initial=execution.initial_residual, final=execution.final_residual,
                                        optimized_vector=execution.optimize_result.x,
                                        jacobian=execution.optimize_result.jac)
                    write_json(fit_dir / "objective_trace.json", list(execution.objective_trace))
                    row["residuals_sha256"] = sha256_file(fit_dir / "residuals.npz")
                    row["objective_trace_sha256"] = sha256_file(fit_dir / "objective_trace.json")
                    write_json(fit_dir / "result.json", row)
                    row["official_outputs"] = save_official_fit(
                        fit_dir / "official", problem, execution,
                        title=f"{fit_id}: synthetic oracle-assisted single-parameter check",
                        noiseless=clean_check)
                    relative_error = abs(getattr(fitted, parameter) - truth_values[parameter]) / abs(truth_values[parameter])
                    threshold = study.criteria["noiseless_single_parameter_relative_error_max"] if clean_check else study.criteria["physical_relative_error_max"]
                    row["recovery_check"] = {"relative_error": relative_error, "threshold": threshold,
                                             "passed": relative_error <= threshold,
                                             "scope": "noiseless implementation sanity" if clean_check else "one oracle-assisted seed; not multi-seed recovery"}
                    row["status"] = "completed"
                    if clean_check and (not row["recovery_check"]["passed"] or not bool(execution.optimize_result.success)):
                        sanity_passed = False
                except Exception as error:
                    row["status"] = "failed"
                    row["failure"] = {"type": type(error).__name__, "message": str(error)}
                    (fit_dir / "failure.txt").write_text(traceback.format_exc())
                    if problem is not None:
                        row["optimization_invoked"] = problem.optimization_invoked
                        record["identification_performed"] |= problem.optimization_invoked
                        row["partial_call_counts"] = problem.call_counts
                        write_json(fit_dir / "partial_objective_trace.json", problem.last_trace)
                        row["partial_objective_trace_sha256"] = sha256_file(fit_dir / "partial_objective_trace.json")
                    if clean_check:
                        sanity_passed = False
                write_json(fit_dir / "result.json", row)
                record["fits"].append(row)
                write_json(output / "run.json", record)
        completed = [row for row in record["fits"] if row["status"] == "completed"]
        record["registered_evidence_fit_count"] = sum(row["registered_evidence_fit"] for row in completed)
        record["sanity_fit_count"] = sum(row["noiseless_sanity"] for row in completed)
        record["registered_matrix_total"] = 82
        record["gate"] = {"noiseless_correctness_passed": sanity_passed,
                          "all_four_fit_and_official_report_paths_completed": len(completed) == 4,
                          "single_seed_recovery_checks_passed": len(completed) == 4 and all(row["recovery_check"]["passed"] for row in completed),
                          "joint_recovery_and_heldout_improvement_evaluated": False}
        record["status"] = "passed" if len(completed) == 4 and sanity_passed and all(row["recovery_check"]["passed"] for row in completed) else "failed"
    except Exception as error:
        record["status"] = "failed"
        record["failure"] = {"type": type(error).__name__, "message": str(error)}
        (output / "failure.txt").write_text(traceback.format_exc())
    record["wall_seconds"] = perf_counter() - started
    write_json(output / "run.json", record)
    print(f"Milestone 3 {record['status']}: {output.relative_to(ROOT)}")
    print(f"Registered completed fits: {record.get('registered_evidence_fit_count', 0)}/82; joint recovery remains unrun.")
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
