"""Fit and seal the three registered joint starts without physical truth.

The interface accepts only the sanitized training view, nominal mechanics,
registered options, and truth-free provenance. It never opens an evaluation
dataset or truth file. Truth-based evaluation belongs in a separate artifact
after ``selection.json`` has been written.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import pickle
from typing import Any

import numpy as np

from mujoco_sysid_lab.dataset import TRAIN_ARRAY_NAMES, TRAIN_TRAJECTORIES, TrainingDataset, array_sha256
from mujoco_sysid_lab.experiment_config import StudyConfig, config_to_dict
from mujoco_sysid_lab.identification import _validate_training, build_problem
from mujoco_sysid_lab.provenance import sha256_file, write_json


START_IDS = ("S0", "S1", "S2")
FREE_PARAMETERS = ("mass_kg", "damping_nm_s_per_rad")
_FORBIDDEN_PROVENANCE_KEYS = frozenset({
    "truth_config", "ground_truth", "ground_truth_config", "truth_generator_config",
    "generation_provenance", "clean_qpos", "clean_qvel", "latent_qpos", "latent_qvel",
    "truth", "truth_values", "truth_model", "ground_truth_parameters",
})


def _sha256_string(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _validate_provenance(provenance: Mapping[str, Any], model_path: Path) -> dict:
    if not isinstance(provenance, Mapping):
        raise ValueError("joint provenance must be an object")

    def inspect(value):
        if isinstance(value, Mapping):
            if set(value) & _FORBIDDEN_PROVENANCE_KEYS:
                raise ValueError("joint core provenance must exclude physical truth, latent arrays, and generator metadata")
            for item in value.values():
                inspect(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                inspect(item)

    inspect(provenance)
    try:
        record = json.loads(json.dumps(provenance, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("joint provenance must be finite strict JSON") from error
    required = {"mujoco_version", "python", "environment", "git", "command", "config",
                "model", "package_versions", "source_sha256"}
    if not required <= set(record):
        raise ValueError("joint provenance is missing required environment/config/source fields")
    for name in ("python", "environment", "git", "command", "config", "model", "package_versions", "source_sha256"):
        if not isinstance(record[name], dict):
            raise ValueError(f"joint provenance {name} must be an object")
    commit = record["git"].get("commit")
    if (not isinstance(commit, str) or len(commit) not in (40, 64)
            or any(c not in "0123456789abcdef" for c in commit)
            or type(record["git"].get("dirty")) is not bool):
        raise ValueError("joint provenance requires real Git commit identity and dirty state")
    if (not _sha256_string(record["config"].get("sha256"))
            or not isinstance(record["config"].get("content"), dict)
            or record["model"].get("sha256") != sha256_file(model_path)):
        raise ValueError("joint provenance requires exact config and matching model hash")
    if not record["command"].get("argv") or not record["python"].get("version"):
        raise ValueError("joint provenance requires exact command and Python version")
    if (not record["environment"].get("cpu") or not record["environment"].get("os")
            or record["mujoco_version"] != "3.15.0"
            or record["package_versions"].get("mujoco") != "3.15.0"):
        raise ValueError("joint provenance requires the pinned library and CPU/OS environment")
    return record


def _selection_eligibility(row: Mapping[str, Any]) -> tuple[bool, str | None]:
    if row.get("optimization_returned") is not True:
        return False, "The official optimization did not return a numerical fit."
    objective = row.get("final_objective")
    if (isinstance(objective, (bool, np.bool_)) or not isinstance(objective, (int, float))
            or not math.isfinite(objective) or objective < 0):
        return False, "A finite nonnegative final training objective is unavailable."
    try:
        vector = np.asarray(row.get("optimized_vector"), dtype=np.float64)
    except (TypeError, ValueError):
        return False, "A finite two-parameter vector is unavailable."
    if vector.shape != (2,) or not np.isfinite(vector).all():
        return False, "A finite two-parameter vector is unavailable."
    return True, None


def select_training_winner(rows: Sequence[Mapping[str, Any]]) -> dict:
    """Choose minimum finite training cost, even for nonconverged/report-failed fits.

    Exact numerical ties prefer S0, then S1, then S2. There is no near-tie
    tolerance. Input order cannot change the tie rule. Physical errors and
    held-out metrics are neither read nor accepted as selection criteria.
    """
    if (not isinstance(rows, Sequence) or isinstance(rows, (str, bytes))
            or len(rows) != len(START_IDS) or any(not isinstance(row, Mapping) for row in rows)):
        raise ValueError("selection requires exactly three registered start rows")
    indexed = {row.get("start_id"): row for row in rows}
    if len(indexed) != 3 or set(indexed) != set(START_IDS):
        raise ValueError("selection requires unique S0/S1/S2 start identities")
    candidates = []
    eligibility = []
    for order, start_id in enumerate(START_IDS):
        row = indexed[start_id]
        if (row.get("fit_id") != f"B_seed101_{start_id}" or row.get("seed") != 101
                or row.get("cell") != "B"):
            raise ValueError("selection fit identity disagrees with registered B seed-101 starts")
        eligible, reason = _selection_eligibility(row)
        eligibility.append({"fit_id": row["fit_id"], "start_id": start_id,
                            "eligible": eligible, "reason": reason,
                            "optimizer_success": row.get("optimizer_success"),
                            "report_status": row.get("report_status")})
        if eligible:
            candidates.append((float(row["final_objective"]), order, row))
    winner = min(candidates, key=lambda candidate: candidate[:2])[2] if candidates else None
    selection = {
        "schema_version": "mujoco-sysid-lab.training-selection.v1",
        "status": "selected" if winner else "all_invalid",
        "selected_fit_id": winner["fit_id"] if winner else None,
        "selected_start_id": winner["start_id"] if winner else None,
        "selected_vector": list(winner["optimized_vector"]) if winner else None,
        "selected_final_objective": float(winner["final_objective"]) if winner else None,
        "selected_result": deepcopy(winner.get("result")) if winner else None,
        "free_parameter_order": list(FREE_PARAMETERS),
        "criterion": "lowest finite final training objective including nonconverged numerical fits",
        "tie_rule": "exact equality only; S0 then S1 then S2",
        "convergence_and_reporting_are_selection_filters": False,
        "physical_truth_evaluated": False, "heldout_evaluated": False,
        "candidate_eligibility": eligibility,
    }
    json.dumps(selection, allow_nan=False)
    return selection


def _file_ref(path: Path, root: Path) -> dict:
    return {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}


def _save_inputs(training: TrainingDataset, directory: Path) -> dict:
    hashes = {}
    for sequence in training.sequences:
        arrays = {name: getattr(sequence, name) for name in TRAIN_ARRAY_NAMES}
        np.savez_compressed(directory / f"{sequence.trajectory_name}_inputs.npz", **arrays)
        hashes[sequence.trajectory_name] = {name: array_sha256(array) for name, array in arrays.items()}
    return hashes


def run_joint_starts(training: TrainingDataset, study: StudyConfig, model_path: Path,
                     output: Path, provenance: dict) -> dict:
    """Run and seal B seed 101 S0/S1/S2; return rows and immutable selection.

    A new root may already contain orchestration provenance/registration files.
    Worker output names must not exist. Existing evidence is never overwritten.
    Reduced horizons are labeled verification; the evidence runner checks the
    registered full horizons before invoking this worker.
    """
    from mujoco_sysid_lab.fit_reporting import save_official_fit, summarize_optimization

    if type(training) is not TrainingDataset or training.seed != 101:
        raise ValueError("Milestone 4 requires sanitized registered seed-101 training inputs")
    if tuple(sequence.trajectory_name for sequence in training.sequences) != TRAIN_TRAJECTORIES:
        raise ValueError("joint fitting requires exactly the two registered training trajectories")
    _validate_training(training, study.nominal, study)
    record = _validate_provenance(provenance, model_path)
    for start_id in START_IDS:
        if start_id not in study.starts:
            raise ValueError("registered joint initialization is missing")
    target = Path(output)
    if target.is_symlink() or (target.exists() and not target.is_dir()):
        raise ValueError("joint output must be a regular run directory")
    for name in ("fits", "training_fits.json", "selection.json"):
        if (target / name).exists() or (target / name).is_symlink():
            raise ValueError("Joint output already exists; preserve evidence and choose a new run directory")
    target.mkdir(parents=True, exist_ok=True)
    (target / "fits").mkdir(exist_ok=False)
    full_horizon = all(math.isclose(float(sequence.timestamps[-1]),
                                   study.trajectories[sequence.trajectory_name].duration_s,
                                   rel_tol=0, abs_tol=1e-12)
                       for sequence in training.sequences)
    scope = "registered_B_seed101" if full_horizon else "reduced_duration_verification"
    rows = []
    for start_id in START_IDS:
        fit_id = f"B_seed101_{start_id}"
        directory = target / "fits" / fit_id
        directory.mkdir(exist_ok=False)
        start = study.starts[start_id]
        start_values = {"mass_kg": start.mass_kg, "damping_nm_s_per_rad": start.damping_nm_s_per_rad}
        row = {
            "schema_version": "mujoco-sysid-lab.joint-training-fit.v1",
            "fit_id": fit_id, "cell": "B", "seed": 101, "start_id": start_id,
            "fit_scope": "joint_physical", "run_purpose": scope,
            "registered_evidence_fit": full_horizon,
            "training_ids": list(TRAIN_TRAJECTORIES), "free_parameters": list(FREE_PARAMETERS),
            "starting_values": start_values, "initial_physics": config_to_dict(study.nominal),
            "optimizer_options": config_to_dict(study.optimizer),
            "oracle_assisted": False, "physical_truth_evaluated": False, "heldout_evaluated": False,
            "truth_visibility": "No physical truth or latent/held-out arrays supplied to this worker.",
            "optimization_attempted": False, "optimization_invoked": False,
            "optimization_returned": False, "optimizer_success": None,
            "optimized_vector": None, "final_objective": None,
            "report_status": "not_run", "status": "prepared", "provenance": record,
        }
        problem = execution = None
        log_path = directory / "optimizer.log"
        try:
            row["input_array_sha256"] = _save_inputs(training, directory)
            row["initial_physics"].update(start_values)
            row["optimization_attempted"] = True
            write_json(directory / "result.json", row)
            problem = build_problem(training, model_path, study.nominal, study, FREE_PARAMETERS, start_values)
            row["resolved_optimizer_options"] = {
                name: value.tolist() if isinstance(value, np.ndarray) else value
                for name, value in problem.optimizer_options.items()
            }
            with log_path.open("w", encoding="utf-8") as log, redirect_stdout(log), redirect_stderr(log):
                execution = problem.run()
            # Persist the numerical fit before any optional diagnostic/report
            # work; report failure must not remove it from training selection.
            vector = execution.optimized_params.as_vector()
            objective = float(.5 * execution.final_residual @ execution.final_residual)
            if not np.isfinite(vector).all() or not np.isfinite(objective):
                raise ValueError("official optimization returned non-finite numerical fit")
            row.update(optimization_returned=True, optimized_vector=vector.tolist(),
                       final_objective=objective, optimizer_success=bool(execution.optimize_result.success))
            native_path = directory / "native_result.pkl"
            with native_path.open("wb") as stream:
                pickle.dump(execution.optimize_result, stream)
            arrays = {"initial": execution.initial_residual, "final": execution.final_residual,
                      "optimized_vector": vector, "jacobian": np.asarray(execution.optimize_result.jac)}
            np.savez_compressed(directory / "residuals.npz", **arrays)
            row["native_result"] = _file_ref(native_path, target)
            row["residuals"] = _file_ref(directory / "residuals.npz", target)
            write_json(directory / "parameters.json", {
                "free_parameter_order": list(FREE_PARAMETERS), "initial": problem.params.as_vector().tolist(),
                "optimized": vector.tolist(), "lower": problem.params.get_bounds()[0].tolist(),
                "upper": problem.params.get_bounds()[1].tolist(),
            })
            row["parameters"] = _file_ref(directory / "parameters.json", target)
            row["optimization_summary"] = summarize_optimization(
                problem, execution, bound_hit_fraction=study.metrics["bound_hit_fraction"],
            )
            row["report_status"] = "started"
            row["official_outputs"] = save_official_fit(
                directory / "official", problem, execution,
                title=f"{fit_id}: synthetic joint mass and damping identification",
                fit_scope="joint_physical",
            )
            row["report_status"] = "completed"
            row["status"] = "completed"
        except Exception as error:
            row["failure"] = {"type": type(error).__name__, "message": str(error)}
            row["status"] = "post_optimization_failed" if execution is not None else "optimization_failed"
            if row["report_status"] == "started":
                row["report_status"] = "failed"
        finally:
            row["optimization_invoked"] = bool(problem and problem.optimization_invoked)
            trace = list(execution.objective_trace) if execution else (problem.last_trace if problem else [])
            counts = execution.call_counts if execution else (problem.call_counts if problem else None)
            write_json(directory / "objective_trace.json", trace)
            row["objective_trace"] = _file_ref(directory / "objective_trace.json", target)
            row["call_counts"] = deepcopy(counts)
            if log_path.exists():
                row["optimizer_log"] = _file_ref(log_path, target)
            row["selection_eligible"], row["selection_ineligible_reason"] = _selection_eligibility(row)
            write_json(directory / "result.json", row)
        index_row = dict(row, result=_file_ref(directory / "result.json", target))
        rows.append(index_row)
    input_hashes = rows[0].get("input_array_sha256", {})
    serialized_input = json.dumps(input_hashes, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    training_identity = {"seed": 101, "trajectory_ids": list(TRAIN_TRAJECTORIES),
                         "array_sha256": input_hashes, "combined_array_sha256": hashlib.sha256(serialized_input).hexdigest()}
    index = {"schema_version": "mujoco-sysid-lab.training-fits.v1", "run_purpose": scope,
             "fits": rows, "training_identity": training_identity, "provenance": record,
             "physical_truth_evaluated": False, "heldout_evaluated": False}
    write_json(target / "training_fits.json", index)
    selection = select_training_winner(rows)
    selection.update({
        "run_purpose": scope, "training_identity": training_identity,
        "dataset_reference": deepcopy(record.get("dataset_input")),
        "config_reference": {"path": record["config"].get("path"), "sha256": record["config"]["sha256"]},
        "source_reference": deepcopy(record["git"]),
        "training_fits": _file_ref(target / "training_fits.json", target),
        "provenance": record,
        "sealed_before_physical_truth_evaluation": True,
    })
    write_json(target / "selection.json", selection)
    return {"fits": rows, "selection": selection,
            "selection_sha256": sha256_file(target / "selection.json"),
            "training_fits_sha256": sha256_file(target / "training_fits.json"),
            "status": "completed" if all(row["status"] == "completed" for row in rows) else "incomplete"}
