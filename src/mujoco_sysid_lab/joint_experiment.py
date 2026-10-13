"""Milestone 4 orchestration: seal training selection before physical evaluation."""
from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import subprocess
import sys
from time import perf_counter
import traceback

import numpy as np

from mujoco_sysid_lab.dataset import array_sha256, load_training_sequences
from mujoco_sysid_lab.evaluation import aggregate_metrics, evaluate_trajectory, relative_improvement
from mujoco_sysid_lab.experiment_config import load_ground_truth, load_study_config
from mujoco_sysid_lab.fit_reporting import summarize_fit
from mujoco_sysid_lab.identification import FitExecution, build_problem
from mujoco_sysid_lab.provenance import collect_provenance, sha256_file, snapshot_sources, write_json
from mujoco_sysid_lab.simulation import compile_model, simulate


def check_registration(config: Path, truth: Path, registration: Path) -> dict:
    """Check the frozen truth file as opaque bytes; do not deserialize its physics."""
    record = json.loads(registration.read_text())
    if record.get("schema_version") != "mujoco-sysid-lab.registration.v1":
        raise ValueError("Unsupported preregistration schema")
    if (sha256_file(config) != record["evidence_config_sha256"]
            or json.loads(config.read_text()) != record["evidence_config"]):
        raise ValueError("Frozen evidence configuration changed")
    if sha256_file(truth) != record["ground_truth_config_sha256"]:
        raise ValueError("Frozen ground-truth file hash changed")
    return record


def evaluate_training(training, physical, study, model_path: Path) -> tuple[dict, dict]:
    model = compile_model(model_path, physical)
    rows, predictions = [], {}
    for sequence in training.sequences:
        predicted = simulate(model, sequence.timestamps, sequence.controls,
                             initial_qpos=float(sequence.initial_state[1]),
                             initial_qvel=float(sequence.initial_state[2]))
        predictions[sequence.trajectory_name] = predicted
        rows.append({"trajectory": sequence.trajectory_name, "split": "train",
                     **evaluate_trajectory(sequence.timestamps, sequence.measured_qpos, sequence.measured_qvel,
                         predicted.qpos, predicted.qvel, study.sensor_sigmas["qpos"], study.sensor_sigmas["qvel"],
                         window_s=study.evaluation_window_for(sequence.trajectory_name))})
    return aggregate_metrics(rows), predictions


def run_milestone4(root: Path, args, output: Path) -> int:
    """Run three registered starts, then inspect truth without altering selection."""
    from mujoco_sysid_lab.joint_identification import run_joint_starts
    started = perf_counter()
    config, registration_path, model_path = root / args.config, root / args.registration, root / "model/pendulum.xml"
    study = load_study_config(config)
    registration = check_registration(config, study.truth_config_path, registration_path)
    executable_path = Path(sys.executable).absolute()
    executable = (executable_path.relative_to(root).as_posix()
                  if executable_path.is_relative_to(root) else str(executable_path))
    provenance = collect_provenance(root, config, model_path, [executable, "scripts/run_experiment.py", *sys.argv[1:]])
    if provenance["mujoco_version"] != "3.15.0" or provenance["python"]["version"] != "3.12.5":
        raise ValueError("Restore registered MuJoCo 3.15.0 / Python 3.12.5 environment")
    provenance.update(seed=101, milestone=4, scenario="matched", created_at_utc=datetime.now(timezone.utc).isoformat(),
                      preregistration={"path": registration_path.relative_to(root).as_posix(), "sha256": sha256_file(registration_path)},
                      ground_truth_file_sha256=sha256_file(study.truth_config_path))
    output.mkdir(parents=True, exist_ok=False)
    if provenance["git"]["dirty"]:
        provenance["source_snapshot"] = snapshot_sources(root, output / "source_snapshot")
    record = {"schema_version": "mujoco-sysid-lab.milestone4.v1", "status": "running", "provenance": provenance,
              "scope": "Seed-101 joint physical fitting from three starts; training-only selection and evaluation.",
              "oracle_assisted": False, "heldout_evaluated": False, "phase_events": []}

    def phase(name):
        record["phase_events"].append({"name": name, "at_utc": datetime.now(timezone.utc).isoformat()})
        write_json(output / "run.json", record)

    write_json(output / "registration.json", registration)
    phase("registration_checked_without_deserializing_truth_config")
    try:
        dataset = (root / args.dataset).resolve() if args.dataset else root / "artifacts/milestone2/20261012_seed101_verified/dataset"
        if args.dataset is None and not dataset.exists():
            generated = output / "dataset_input"
            command = [executable, "scripts/generate_data.py", "--config", args.config, "--output", generated.relative_to(root).as_posix()]
            with (output / "dataset_generation.log").open("w") as log:
                subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
            dataset = generated / "dataset"
        manifest = dataset / "manifest.json" if dataset.is_dir() else dataset
        header = json.loads(manifest.read_text())
        if (not manifest.resolve().is_relative_to(root / "artifacts") or header["seed"] != 101 or header["scenario"] != "matched"
                or header["provenance"]["config"]["sha256"] != sha256_file(config)
                or header["provenance"]["model"]["sha256"] != sha256_file(model_path)
                or header["metadata"]["truth_config_sha256"] != sha256_file(study.truth_config_path)):
            raise ValueError("Dataset must match the authored model, frozen config/truth, and seed 101")
        training = load_training_sequences(manifest)
        for sequence in training.sequences:
            count = int(round(study.trajectories[sequence.trajectory_name].duration_s / study.timestep_s)) + 1
            if sequence.timestamps.size != count:
                raise ValueError("Registered joint evidence requires complete eight-second trajectories")
        dataset_ref = {"path": manifest.relative_to(root).as_posix(), "sha256": sha256_file(manifest),
                       "training_ids": [sequence.trajectory_name for sequence in training.sequences], "heldout_arrays_loaded": False}
        record["dataset_input"] = dataset_ref
        # The worker receives neither the full manifest nor registration's truth
        # snapshot. Only opaque input hashes are attached to its provenance.
        worker_provenance = {**provenance, "dataset_input": dataset_ref}
        phase("sanitized_training_inputs_loaded")
        fitted = run_joint_starts(training, study, model_path, output, worker_provenance)
        record["training_fits"] = fitted
        fit_counts = {
            "registered_slots": 3,
            "optimization_attempted": sum(bool(row.get("optimization_attempted")) for row in fitted["fits"]),
            "optimization_invoked": sum(bool(row.get("optimization_invoked")) for row in fitted["fits"]),
            "numerical_fits_returned": sum(bool(row.get("optimization_returned")) for row in fitted["fits"]),
            "fully_saved_reported_fits": sum(row.get("status") == "completed" for row in fitted["fits"]),
            "nonconverged_numerical_fits": sum(row.get("optimization_returned") is True and row.get("optimizer_success") is False for row in fitted["fits"]),
            "incomplete_fit_or_report_rows": sum(row.get("status") != "completed" for row in fitted["fits"]),
        }
        record["fit_counts"] = fit_counts
        record["registered_evidence_fits_this_milestone"] = fit_counts["fully_saved_reported_fits"]
        record["cumulative_registered_evidence_fit_count"] = 2 + fit_counts["fully_saved_reported_fits"]
        selection_path = output / "selection.json"
        selection_hash = sha256_file(selection_path)
        selection = json.loads(selection_path.read_text())
        record["selection"] = selection
        record["selection_sha256"] = selection_hash
        phase("three_start_attempts_complete_and_training_selection_sealed")
        selected_id = selection.get("selected_fit_id")
        if selected_id is None:
            raise ValueError("No finite training candidate; physical evaluation was not run")
        phase("post_selection_truth_evaluation_started")
        truth = load_ground_truth(study.truth_config_path, study)
        truth_values = {"mass_kg": truth.model.mass_kg, "damping_nm_s_per_rad": truth.model.damping_nm_s_per_rad}
        nominal_metrics, nominal_predictions = evaluate_training(training, study.nominal, study, model_path)
        oracle_metrics, oracle_predictions = evaluate_training(training, truth.model, study, model_path)
        evaluations, predictions = [], {}
        for row in fitted["fits"]:
            if not row.get("optimization_returned", False):
                evaluations.append({"fit_id": row["fit_id"], "status": "unavailable", "reason": "Optimization did not return a finite candidate"})
                continue
            fit_dir = output / "fits" / row["fit_id"]
            for key in ("result", "native_result", "residuals", "objective_trace", "parameters"):
                reference = row[key]
                if sha256_file(output / reference["path"]) != reference["sha256"]:
                    raise ValueError(f"Sealed training artifact changed: {key}")
            with (fit_dir / "native_result.pkl").open("rb") as handle:
                native = pickle.load(handle)  # Authored, freshly saved official result.
            with np.load(fit_dir / "residuals.npz", allow_pickle=False) as arrays:
                initial, final = arrays["initial"].copy(), arrays["final"].copy()
                np.testing.assert_array_equal(native.fun, final)
                np.testing.assert_array_equal(native.x, arrays["optimized_vector"])
                np.testing.assert_array_equal(native.jac, arrays["jacobian"])
            start = study.starts[row["start_id"]]
            initial_values = {"mass_kg": start.mass_kg, "damping_nm_s_per_rad": start.damping_nm_s_per_rad}
            problem = build_problem(training, model_path, study.nominal, study, tuple(initial_values), initial_values)
            optimized = problem.params.copy()
            optimized.update_from_vector(native.x)
            diagnostics = row["optimization_summary"]
            trace = json.loads((fit_dir / "objective_trace.json").read_text())
            execution = FitExecution(optimized, native, initial, final, [], [], [], [], diagnostics["call_counts"],
                                     tuple(trace), diagnostics["optimizer"]["callback_events"], diagnostics["runtime_seconds"])
            summary = summarize_fit(problem, execution, truth_values, bound_hit_fraction=study.metrics["bound_hit_fraction"])
            physics = replace(study.nominal, **dict(zip(initial_values, native.x.tolist(), strict=True)))
            metrics, prediction = evaluate_training(training, physics, study, model_path)
            metric_objective = sum(item["n_samples"] for item in metrics["trajectory_metrics"]) * metrics["normalized_rmse"] ** 2
            if not np.isclose(metric_objective, native.cost, rtol=1e-9, atol=1e-8):
                raise ValueError("Shared training metrics disagree with official joint objective")
            predictions[row["fit_id"]] = prediction
            evaluations.append({"fit_id": row["fit_id"], "status": "evaluated", "selected": row["fit_id"] == selected_id,
                                "summary": summary, "fitted_physics": asdict(physics), "training_metrics": metrics,
                                "training_improvement_relative_to_nominal": relative_improvement(metrics, nominal_metrics),
                                "metric_objective_consistency": {"passed": True, "shared_metric_objective": metric_objective, "native_cost": float(native.cost)}})
        chosen = next(row for row in evaluations if row.get("selected"))
        evaluation = {"schema_version": "mujoco-sysid-lab.joint-evaluation.v1", "provenance": provenance,
                      "truth_evaluation_only": asdict(truth.model), "selection_sha256": selection_hash,
                      "nominal_training_metrics": nominal_metrics, "oracle_training_metrics": oracle_metrics,
                      "fits": evaluations, "heldout_evaluated": False,
                      "delay_stage_physical_eligibility": {"fit_id": selected_id,
                          "threshold_relative_error": study.criteria["delay_physical_eligibility_relative_error_max"],
                          "eligible": all(row["relative_error"] <= study.criteria["delay_physical_eligibility_relative_error_max"] for row in chosen["summary"]["parameter_rows"]),
                          "does_not_run_delay_identification": True}}
        write_json(output / "evaluation.json", evaluation)
        prediction_dir = output / "training_predictions"
        prediction_dir.mkdir(exist_ok=False)
        descriptors = {}
        for sequence in training.sequences:
            name = sequence.trajectory_name
            arrays = {"timestamps": sequence.timestamps, "controls": sequence.controls,
                      "initial_state": sequence.initial_state, "measured_qpos": sequence.measured_qpos, "measured_qvel": sequence.measured_qvel}
            for label, values in (("nominal", nominal_predictions), ("identified", predictions[selected_id]), ("oracle", oracle_predictions)):
                arrays[f"{label}_qpos"], arrays[f"{label}_qvel"] = values[name].qpos, values[name].qvel
            np.savez_compressed(prediction_dir / f"{name}.npz", **arrays)
            descriptors[name] = {"path": f"training_predictions/{name}.npz", "sha256": sha256_file(prediction_dir / f"{name}.npz"),
                                 "array_sha256": {key: array_sha256(value) for key, value in arrays.items()}}
        if sha256_file(selection_path) != selection_hash:
            raise ValueError("Post-selection physical evaluation mutated training selection")
        record.update(evaluation={"path": "evaluation.json", "sha256": sha256_file(output / "evaluation.json")},
                      training_predictions=descriptors, evaluation_results=evaluation,
                      registered_matrix_total=82,
                      gate={"training_selection_unchanged_after_truth_evaluation": True,
                            "all_three_fit_reports_completed": all(row.get("status") == "completed" for row in fitted["fits"]),
                            "shared_metric_objectives_verified": True,
                            "heldout_improvement_and_five_seed_targets_evaluated": False})
        record["status"] = "passed" if record["gate"]["all_three_fit_reports_completed"] else "failed"
        phase("post_selection_training_evaluation_complete")
    except Exception as error:
        record["status"] = "failed"
        record["failure"] = {"type": type(error).__name__, "message": str(error)}
        (output / "failure.txt").write_text(traceback.format_exc())
    record["wall_seconds"] = perf_counter() - started
    write_json(output / "run.json", record)
    print(f"Milestone 4 {record['status']}: {output.relative_to(root)}")
    print("Three starts retained; held-out prediction remains unscored.")
    return 0 if record["status"] == "passed" else 1
