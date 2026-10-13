"""Registered CPU study: truth-free fitting, sealed selection, then evaluation."""
from __future__ import annotations

from contextlib import redirect_stdout, redirect_stderr
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import pickle
import subprocess
import sys
import tarfile
import tempfile
from time import perf_counter
import traceback

import numpy as np

from mujoco_sysid_lab.dataset import TRAIN_ARRAY_NAMES, array_sha256
from mujoco_sysid_lab.experiment_config import config_to_dict, load_study_config, load_ground_truth
from mujoco_sysid_lab.identification import build_problem
from mujoco_sysid_lab.fit_reporting import summarize_optimization, save_official_fit
from mujoco_sysid_lab.provenance import collect_provenance, sha256_file, snapshot_sources, write_json


LEGACY_CACHE_RUNS = {
    "unit": "artifacts/milestone3/20261012_seed101_verified",
    "physical": "artifacts/milestone4/20261013_seed101",
    "delay": "artifacts/milestone6/20261013_seed101",
}
_ESTIMATOR_SOURCES = ("src/mujoco_sysid_lab/identification.py", "src/mujoco_sysid_lab/api_check.py",
                      "src/mujoco_sysid_lab/simulation.py", "src/mujoco_sysid_lab/experiment_config.py")


def cache_directory(root: Path, supplied, kind: str) -> Path:
    """Optional historical reuse by default; explicit paths must exist."""
    path = root / (supplied if supplied is not None else LEGACY_CACHE_RUNS[kind])
    if path.name == "run.json":
        path = path.parent
    path = path.resolve()
    if not path.is_relative_to((root / "artifacts").resolve()):
        raise ValueError("Cache runs must be retained evidence under artifacts/")
    if supplied is not None and not (path / "run.json").is_file():
        raise ValueError(f"Explicit {kind} cache run is missing run.json: {path}")
    return path


def verify_cache_source(root: Path, provenance: dict) -> dict:
    """Verify original estimator bytes, allowing audited legacy hook versions.

    Source provenance proves what ran. Compatibility is then checked by the
    frozen model/config/options/training inputs and the shared objective replay,
    rather than pretending old and current reporting/delay source are identical.
    """
    commit = provenance["git"]["commit"]
    ledger = provenance["source_sha256"]
    critical = {*_ESTIMATOR_SOURCES, provenance["config"]["path"],
                provenance["model"]["path"], provenance["lock"]["path"]}
    if None in critical or not critical <= ledger.keys():
        raise ValueError("Cached source provenance lacks required estimator/config/model/lock hashes")
    if any(provenance[name]["sha256"] != ledger[provenance[name]["path"]] for name in ("config", "model", "lock")):
        raise ValueError("Cached config/model/lock metadata disagrees with source ledger")
    if provenance["mujoco_version"] != "3.15.0" or provenance["python"]["version"] != "3.12.5":
        raise ValueError("Cached source violates pinned MuJoCo/Python")
    if any(provenance["package_versions"].get(name) != version for name, version in
           {"mujoco": "3.15.0", "numpy": "2.3.3", "scipy": "1.16.2"}.items()):
        raise ValueError("Cached estimator libraries differ from registered pins")
    if provenance["git"]["dirty"]:
        snapshot = provenance.get("source_snapshot")
        if snapshot is None:
            raise ValueError("Dirty cache requires a reconstructable source snapshot")
        directory = (root / snapshot["path"]).resolve()
        if not directory.is_relative_to((root / "artifacts").resolve()):
            raise ValueError("Cache source snapshot escapes artifacts/")
        manifest_path = directory / "manifest.json"
        if sha256_file(manifest_path) != snapshot["manifest_sha256"]:
            raise ValueError("Cache source snapshot manifest changed")
        manifest = json.loads(manifest_path.read_text())
        patch = directory / manifest["tracked_patch"]["path"]
        if (manifest["base_commit"] != commit or snapshot["base_commit"] != commit
                or manifest["source_sha256"] != ledger or snapshot["source_sha256"] != ledger
                or sha256_file(patch) != manifest["tracked_patch"]["sha256"]
                or sha256_file(patch) != snapshot["tracked_patch_sha256"]):
            raise ValueError("Cache source snapshot/patch ledger changed")
        with tempfile.TemporaryDirectory(prefix="sysid_cache_source_") as temporary:
            tree = Path(temporary).resolve()
            result = subprocess.run(["git", "-C", str(root), "archive", "--format=tar", commit],
                                    check=True, capture_output=True)
            with tarfile.open(fileobj=io.BytesIO(result.stdout)) as archive:
                for member in archive.getmembers():
                    target = (tree / member.name).resolve()
                    if not target.is_relative_to(tree) or not (member.isdir() or member.isfile()):
                        raise ValueError("Unsafe cached Git source archive")
                    if member.isfile():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with archive.extractfile(member) as stream:
                            target.write_bytes(stream.read())
            if patch.stat().st_size:
                subprocess.run(["git", "apply", "--binary", "--whitespace=nowarn", str(patch)],
                               cwd=tree, check=True, capture_output=True)
            for relative in manifest["untracked_sources"]:
                original, target = (directory / "untracked" / relative).resolve(), (tree / relative).resolve()
                if not original.is_relative_to(directory / "untracked") or not target.is_relative_to(tree):
                    raise ValueError("Unsafe cached untracked source path")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(original.read_bytes())
            if any(sha256_file(tree / path) != ledger[path] for path in critical):
                raise ValueError("Reconstructed cached estimator source hash changed")
            if json.loads((tree / provenance["config"]["path"]).read_text()) != provenance["config"]["content"]:
                raise ValueError("Cached exact config content disagrees with original source")
    else:
        for path in critical:
            result = subprocess.run(["git", "-C", str(root), "show", f"{commit}:{path}"],
                                    check=True, capture_output=True)
            if hashlib.sha256(result.stdout).hexdigest() != ledger[path]:
                raise ValueError(f"Cached estimator source hash changed: {path}")
            if path == provenance["config"]["path"] and json.loads(result.stdout) != provenance["config"]["content"]:
                raise ValueError("Cached exact config content disagrees with original source")
    return {"policy": "frozen noise-normalized joint-sensor objective v1; original source bytes verified; objective replay required",
            "source_commit": commit, "source_dirty": provenance["git"]["dirty"],
            "estimator_source_sha256": {path: ledger[path] for path in sorted(critical)},
            "current_source_byte_equality_required": False}


def file_ref(path: Path, root: Path) -> dict:
    return {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}


def training_hashes(training) -> dict:
    return {sequence.trajectory_name: {name: array_sha256(getattr(sequence, name)) for name in TRAIN_ARRAY_NAMES}
            for sequence in training.sequences}


def select_rows(rows: list[dict], free_parameters: tuple[str, ...]) -> dict:
    """Finite final training cost only; stable exact ties in registered order."""
    if (not rows or len({row.get("start_id") for row in rows}) != len(rows)
            or any(row.get("start_id") not in ("S0", "S1", "S2") for row in rows)
            or not free_parameters or len(set(free_parameters)) != len(free_parameters)
            or len({row.get("seed") for row in rows}) != 1 or len({row.get("cell") for row in rows}) != 1
            or any(row.get("fit_id") != f"{row.get('cell')}_seed{row.get('seed')}_{row.get('start_id')}"
                   or tuple(row.get("free_parameters", ())) != free_parameters for row in rows)):
        raise ValueError("Selection requires unique registered start identities and free parameters")
    candidates = []
    for row in rows:
        order = ("S0", "S1", "S2").index(row["start_id"])
        vector = row.get("optimized_vector")
        cost = row.get("final_objective")
        if (row.get("optimization_returned") is True and isinstance(vector, list)
                and len(vector) == len(free_parameters)
                and all(isinstance(value, (float, int)) and not isinstance(value, bool) for value in vector)
                and np.isfinite(vector).all()
                and isinstance(cost, (float, int)) and not isinstance(cost, bool) and np.isfinite(cost) and cost >= 0):
            candidates.append((cost, order, row))
    chosen = min(candidates, key=lambda item: item[:2])[2] if candidates else None
    return {"schema_version": "mujoco-sysid-lab.selection.v2", "selected_fit_id": chosen["fit_id"] if chosen else None,
            "selected_vector": chosen["optimized_vector"] if chosen else None,
            "selected_final_objective": chosen["final_objective"] if chosen else None,
            "free_parameter_order": list(free_parameters), "candidate_fit_ids": [row["fit_id"] for row in rows],
            "criterion": "minimum finite final training cost including nonconverged candidates",
            "tie_rule": "exact equality only; S0 then S1 then S2", "physical_truth_evaluated": False,
            "heldout_evaluated": False}


def fit_one(training, study, model_path: Path, physical, free_parameters: tuple[str, ...], start: dict,
            fit_id: str, cell: str, start_id: str, directory: Path, provenance: dict) -> dict:
    """Estimator worker receives sanitized training and known/nominal physics only."""
    directory.mkdir(parents=True, exist_ok=False)
    row = {"schema_version": "mujoco-sysid-lab.training-fit.v2", "fit_id": fit_id, "cell": cell,
           "seed": training.seed, "start_id": start_id, "free_parameters": list(free_parameters),
           "starting_values": start, "initial_physics": config_to_dict(physical),
           "training_ids": [sequence.trajectory_name for sequence in training.sequences],
           "input_array_sha256": training_hashes(training), "provenance": provenance,
           "physical_truth_evaluated": False, "heldout_evaluated": False,
           "optimization_invoked": False, "optimization_returned": False, "optimizer_success": None,
           "optimized_vector": None, "final_objective": None, "status": "prepared"}
    write_json(directory / "result.json", row)
    problem = execution = None
    try:
        problem = build_problem(training, model_path, physical, study, free_parameters, start,
                                training_ids=tuple(row["training_ids"]))
        row["resolved_optimizer_options"] = {name: value.tolist() if isinstance(value, np.ndarray) else value
                                              for name, value in problem.optimizer_options.items()}
        with (directory / "optimizer.log").open("w") as log, redirect_stdout(log), redirect_stderr(log):
            execution = problem.run()
        row.update(optimization_returned=True, optimizer_success=bool(execution.optimize_result.success),
                   optimized_vector=execution.optimized_params.as_vector().tolist(),
                   final_objective=float(.5 * execution.final_residual @ execution.final_residual))
        with (directory / "native_result.pkl").open("wb") as stream:
            pickle.dump(execution.optimize_result, stream)
        np.savez_compressed(directory / "residuals.npz", initial=execution.initial_residual,
                            final=execution.final_residual, optimized_vector=execution.optimized_params.as_vector(),
                            jacobian=execution.optimize_result.jac)
        write_json(directory / "objective_trace.json", list(execution.objective_trace))
        write_json(directory / "parameters.json", {"free_parameter_order": list(free_parameters),
                   "initial": problem.params.as_vector().tolist(), "optimized": row["optimized_vector"],
                   "lower": problem.params.get_bounds()[0].tolist(), "upper": problem.params.get_bounds()[1].tolist()})
        row["optimization_summary"] = summarize_optimization(problem, execution,
                                                             bound_hit_fraction=study.metrics["bound_hit_fraction"])
        row["official_outputs"] = save_official_fit(directory / "official", problem, execution,
                                                     title=f"{fit_id}: synthetic SysID", fit_scope=("oracle_assisted_single_parameter"
                                                     if cell in ("A1", "A2") else "registered_ablation"))
        row["status"] = "completed"
    except KeyboardInterrupt:
        row["status"] = "interrupted"
        row["failure"] = {"type": "KeyboardInterrupt", "message": "Optimization/reporting interrupted"}
        (directory / "failure.txt").write_text(traceback.format_exc())
        raise
    except Exception as error:
        row["status"] = "post_optimization_failed" if execution is not None else "optimization_failed"
        row["failure"] = {"type": type(error).__name__, "message": str(error)}
        (directory / "failure.txt").write_text(traceback.format_exc())
    finally:
        row["optimization_invoked"] = bool(problem and problem.optimization_invoked)
        row["call_counts"] = execution.call_counts if execution else (problem.call_counts if problem else None)
        if "objective_trace" not in row and not (directory / "objective_trace.json").exists():
            write_json(directory / "objective_trace.json", problem.last_trace if problem else [])
        for name in ("native_result.pkl", "residuals.npz", "parameters.json", "objective_trace.json", "optimizer.log"):
            if (directory / name).is_file():
                row[name.removesuffix(".pkl").removesuffix(".npz").removesuffix(".json").removesuffix(".log")] = {
                    "path": name, "sha256": sha256_file(directory / name)}
        write_json(directory / "result.json", row)
    return row


def validate_cached_identity(row: dict, training, study, physical, free: tuple[str, ...], cell: str, *, old_B: bool = False) -> None:
    start_id = row["start_id"]
    if start_id not in study.starts:
        raise ValueError("Cached start identity is unsupported")
    initialization = config_to_dict(study.starts[start_id])
    expected_start = {name: initialization[name] for name in free}
    if cell == "C_delay_only":
        expected_start = {"delay_s": study.optimizer["delay_only_start_s"]}
    expected_physics = config_to_dict(physical)
    if old_B:
        expected_physics.update(expected_start)
    if (row["cell"] != cell or row["seed"] != training.seed
            or row["fit_id"] != f"{cell}_seed{training.seed}_{start_id}"
            or tuple(row["free_parameters"]) != free or row["starting_values"] != expected_start
            or row["initial_physics"] != expected_physics
            or row["training_ids"] != [sequence.trajectory_name for sequence in training.sequences]
            or row["input_array_sha256"] != training_hashes(training)):
        raise ValueError("Cached fit identity, start, physical constants, or training arrays differ")
    validate_cached_options(row, study, free)


def validate_cached_options(row: dict, study, free: tuple[str, ...]) -> None:
    """Recorded effective solver settings must agree with the frozen config."""
    if "optimizer_options" in row and row["optimizer_options"] != config_to_dict(study.optimizer):
        raise ValueError("Cached optimizer options differ")
    if "resolved_optimizer_options" in row:
        bounds = {"mass_kg": study.mass_bounds_kg, "damping_nm_s_per_rad": study.damping_bounds_nm_s_per_rad,
                  "delay_s": study.delay_bounds_s}
        expected = {name: study.optimizer[name] for name in ("method", "loss", "max_iters", "ftol", "xtol", "gtol", "diff_step")}
        expected["x_scale"] = [bounds[name].upper - bounds[name].lower for name in free]
        if row["resolved_optimizer_options"] != expected:
            raise ValueError("Cached resolved optimizer options differ")


def enrich_parameters(row: dict, truth_values: dict) -> list[dict]:
    rows = []
    for parameter in row.get("optimization_summary", {}).get("parameter_rows", []):
        value = truth_values[parameter["name"]]
        error = abs(parameter["estimate"] - value)
        rows.append({**parameter, "truth": value, "absolute_error": error,
                     "relative_error": error / abs(value) if value else None,
                     "relative_error_reason": None if value else "Zero truth reference"})
    return rows


def cached_B(root: Path, training, study, model_path: Path, *, cache_run=None) -> tuple[list[dict], Path] | None:
    source = cache_directory(root, cache_run, "physical")
    if training.seed != 101 or not (source / "run.json").is_file():
        return None
    record = json.loads((source / "run.json").read_text())
    compatibility = verify_cache_source(root, record["provenance"])
    if (record["provenance"]["config"]["sha256"] != sha256_file(root / "configs/evidence_v1.json")
            or record["provenance"]["model"]["sha256"] != sha256_file(model_path)):
        raise ValueError("Existing B evidence has different config or model")
    indexed = record["training_fits"]["fits"]
    if len(indexed) != 3 or {row["start_id"] for row in indexed} != {"S0", "S1", "S2"}:
        raise ValueError("Cached B must preserve all three registered starts")
    rows = []
    for indexed_row in indexed:
        reference = indexed_row["result"]
        result_path = source / reference["path"]
        if sha256_file(result_path) != reference["sha256"]:
            raise ValueError("Cached numerical fit changed")
        row = json.loads(result_path.read_text())
        if any(indexed_row.get(name) != value for name, value in row.items()):
            raise ValueError("Cached fit index disagrees with immutable numerical record")
        row["result"] = reference
        validate_cached_identity(row, training, study, study.nominal,
                                 ("mass_kg", "damping_nm_s_per_rad"), "B", old_B=True)
        if row["optimizer_options"] != config_to_dict(study.optimizer):
            raise ValueError("Cached B optimizer options differ")
        row["cache_compatibility"] = compatibility
        rows.append(row)
    return rows, source


def cached_milestone6(root: Path, cell: str, training, study, model_path: Path, physical, free: tuple[str, ...], *, cache_run=None) -> tuple[list[dict], Path] | None:
    source = cache_directory(root, cache_run, "delay")
    if training.seed != 101 or cell == "B" or not (source / "run.json").is_file():
        return None
    run = json.loads((source / "run.json").read_text())
    compatibility = verify_cache_source(root, run["provenance"])
    if (run["provenance"]["config"]["sha256"] != sha256_file(root / "configs/evidence_v1.json")
            or run["provenance"]["model"]["sha256"] != sha256_file(model_path)):
        raise ValueError("Milestone 6 cache config changed")
    group = next((item for item in run["groups"] if item["group_id"] == f"{cell}_seed101"), None)
    if group is None:
        return None
    rows = []
    for retained in run["fits"]:
        if retained["fit_id"] not in group["fits"]:
            continue
        reference = retained["source_result"]
        path = root / reference["path"]
        if sha256_file(path) != reference["sha256"]:
            raise ValueError("Milestone 6 cached result changed")
        row = json.loads(path.read_text())
        validate_cached_identity(row, training, study, physical, free, cell)
        row["cache_compatibility"] = compatibility
        rows.append(row)
    expected_starts = {"S0"} if cell == "C_delay_only" else {"S0", "S1", "S2"}
    if len(rows) != len(expected_starts) or {row["start_id"] for row in rows} != expected_starts:
        raise ValueError("Milestone 6 cache omitted or duplicated registered starts")
    return rows, source / "fits"


def cached_units(root: Path, training, study, model_path: Path, *, cache_run=None) -> list[dict] | None:
    source = cache_directory(root, cache_run, "unit")
    if not (source / "run.json").is_file():
        return None
    path = source / "run.json"
    run = json.loads(path.read_text())
    compatibility = verify_cache_source(root, run["provenance"])
    if (run["provenance"]["config"]["sha256"] != sha256_file(root / "configs/evidence_v1.json")
            or run["provenance"]["model"]["sha256"] != sha256_file(model_path)):
        raise ValueError("Cached unit config differs")
    rows = []
    for row in run["fits"]:
        if row["noiseless_sanity"]:
            continue
        if row["input_array_sha256"] != training_hashes(training):
            raise ValueError("Cached unit training differs")
        optimization_source = verify_cache_source(root, row.get("optimization_provenance", row["provenance"]))
        parameter_rows = row["summary"]["parameter_rows"]
        validate_cached_options(row, study, tuple(row["free_parameters"]))
        rows.append({**row, "optimization_summary": row["summary"], "parameter_rows": parameter_rows,
                     "optimization_invoked": True, "optimization_returned": True,
                     "optimizer_success": row["summary"]["optimizer"]["success"],
                     "optimized_vector": [parameter["estimate"] for parameter in parameter_rows],
                     "final_objective": row["summary"]["objective"]["final"], "reused": True,
                     "source_result": file_ref(path, root), "source_fit_id": row["fit_id"],
                     "cache_compatibility": {"report_source": compatibility, "optimization_source": optimization_source},
                     "source_official_directory": (source / "fits" / row["fit_id"] / "official").relative_to(root).as_posix()})
    return rows


def run_study(root: Path, args, output: Path, *, through: int = 9) -> int:
    from mujoco_sysid_lab.advanced_data import create_scenario, write_scenario
    from mujoco_sysid_lab.full_evaluation import evaluate_model, phase_lag_diagnostics
    from mujoco_sysid_lab.joint_experiment import check_registration
    from mujoco_sysid_lab.evaluation import relative_improvement
    from mujoco_sysid_lab.api_check import check_api

    started = perf_counter()
    study = load_study_config(root / args.config)
    model_path = root / "model/pendulum.xml"
    registration = check_registration(root / args.config, study.truth_config_path, root / args.registration)
    provenance = collect_provenance(root, root / args.config, model_path,
                                    [".venv/bin/python", "scripts/run_experiment.py", *sys.argv[1:]])
    if provenance["mujoco_version"] != "3.15.0" or provenance["python"]["version"] != "3.12.5":
        raise ValueError("Full evidence requires pinned MuJoCo/Python")
    output.mkdir(parents=True, exist_ok=False)
    if provenance["git"]["dirty"]:
        provenance["source_snapshot"] = snapshot_sources(root, output / "source_snapshot")
    write_json(output / "registration.json", registration)
    write_json(output / "provenance.json", provenance)
    write_json(output / "run.json", {"schema_version": "mujoco-sysid-lab.full-run.v1", "status": "running",
               "phase": "api_contract", "provenance": provenance, "fits": [], "groups": [],
               "datasets": {}, "milestones": [], "wall_seconds": perf_counter() - started})
    timing_contract = check_api(model_path, study.nominal)
    write_json(output / "api_contract.json", {"provenance": provenance, "checks": timing_contract,
                                              "fixed_grid_delay_crop_regression": "Locally verified by tests/test_sysid_contract.py; official resampler on exact fixed grid"})
    # Truth is confined to generator/evaluator orchestration, never fit_one.
    truth = load_ground_truth(study.truth_config_path, study)
    all_rows, groups, datasets, milestones = [], [], {}, []
    selected_B = {}
    seeds = (101,) if through == 6 else study.noise_seeds
    cells = ("B", "delay_excluded", "C_delay_only", "C_joint") if through == 6 else (
        "B", "weak", "delay_excluded", "C_delay_only", "C_joint", *(("E",) if through >= 8 else ()))
    specifications = {row["id"]: row for row in study.evidence_matrix}
    progress = {"phase": "prepared", "active_group": None, "active_fit": None, "active_dataset": None}
    fresh = bool(getattr(args, "fresh", False))
    cache_policy = {"fresh": fresh, "runs": {kind: None if fresh else (getattr(args, f"{kind}_cache_run", None)
                    or default) for kind, default in LEGACY_CACHE_RUNS.items()}}

    def checkpoint():
        write_json(output / "run.json", {"schema_version": "mujoco-sysid-lab.full-run.v1", "status": "running",
                   "provenance": provenance, "fits": all_rows, "groups": groups, "datasets": datasets,
                   "milestones": milestones, "progress": progress, "cache_policy": cache_policy,
                   "wall_seconds": perf_counter() - started})

    def retain(rows):
        for row in rows:
            index = next((index for index, existing in enumerate(all_rows) if existing["fit_id"] == row["fit_id"]), None)
            if index is None:
                all_rows.append(row)
            else:
                all_rows[index] = row
        checkpoint()

    def fit_and_retain(training, physical, free, start, fit_id, cell, start_id, directory, fit_provenance):
        progress.update(phase="fitting", active_fit=fit_id)
        prepared = {"fit_id": fit_id, "cell": cell, "seed": training.seed, "start_id": start_id,
                    "status": "prepared", "free_parameters": list(free), "starting_values": start,
                    "optimization_invoked": False, "optimization_returned": False, "reused": False,
                    "provenance": fit_provenance}
        retain([prepared])
        try:
            row = fit_one(training, study, model_path, physical, free, start, fit_id, cell, start_id,
                          directory, fit_provenance)
        except BaseException:
            # The worker's finally block persists flags/partial traces even on
            # interruption. Reflect those exact bytes in the parent journal.
            result = directory / "result.json"
            if result.is_file():
                retain([{**json.loads(result.read_text()), "reused": False,
                         "source_result": file_ref(result, root)}])
            raise
        row = {**row, "reused": False, "source_result": file_ref(directory / "result.json", root)}
        retain([row])
        progress["active_fit"] = None
        return row

    def scenario_for(seed, scenario):
        key = f"{scenario}_seed{seed}"
        if key not in scenario_cache:
            progress.update(phase="dataset_generation", active_dataset=key)
            checkpoint()
            data = create_scenario(study, truth, model_path, seed, scenario)
            directory = output / "datasets" / key
            dataset_provenance = {**provenance, "seed": seed, "scenario": scenario}
            manifest = write_scenario(directory, data, dataset_provenance)
            datasets[key] = file_ref(manifest, root)
            scenario_cache[key] = data
            progress["active_dataset"] = None
            checkpoint()
        return scenario_cache[key]

    scenario_cache = {}
    checkpoint()
    if through >= 7:
        unit_data = scenario_for(101, "matched")
        units = None if fresh else cached_units(root, unit_data.training, study, model_path,
                                                cache_run=getattr(args, "unit_cache_run", None))
        if units is None:
            units = []
            for cell in ("A1", "A2"):
                free = tuple(specifications[cell]["free_parameters"])
                physical = replace(study.nominal, **({"damping_nm_s_per_rad": truth.model.damping_nm_s_per_rad}
                                                       if cell == "A1" else {"mass_kg": truth.model.mass_kg}))
                row = fit_and_retain(unit_data.training, physical, free,
                              {name: getattr(study.starts["S0"], name) for name in free}, f"{cell}_seed101_S0", cell,
                              "S0", output / "fits" / f"{cell}_seed101_S0", {**provenance, "seed": 101, "cell": cell})
                row.update(reused=False, parameter_rows=enrich_parameters(row, {"mass_kg": truth.model.mass_kg,
                             "damping_nm_s_per_rad": truth.model.damping_nm_s_per_rad}),
                           source_result=file_ref(output / "fits" / row["fit_id"] / "result.json", root))
                units.append(row)
        retain(units)
    for cell in cells:
        for seed in seeds:
            matrix = specifications[cell]
            scenario = "weak" if cell == "weak" else matrix["scenario"]
            data = scenario_for(seed, scenario)
            training = data.training
            free = tuple(matrix["free_parameters"])
            group_id = f"{cell}_seed{seed}"
            progress.update(phase="group_preparation", active_group=group_id, active_fit=None)
            checkpoint()
            directory = output / "groups" / group_id
            directory.mkdir(parents=True, exist_ok=False)
            physical = study.nominal
            prerequisite = None
            if cell == "C_delay_only":
                reference = selected_B.get(seed)
                if reference is None or not reference["physical_eligible"]:
                    prerequisite = "Same-seed B physical recovery prerequisite failed"
                else:
                    physical = reference["physical"]
            if cell == "C_joint":
                previous = next((group for group in groups if group["group_id"] == f"C_delay_only_seed{seed}"), None)
                if previous is None or not previous.get("joint_delay_eligible"):
                    prerequisite = "Same-seed delay-only one-timestep prerequisite failed"
            rows, source = [], output / "fits"
            cache = None if prerequisite or fresh else (
                cached_B(root, training, study, model_path, cache_run=getattr(args, "physical_cache_run", None)) if cell == "B" else
                cached_milestone6(root, cell, training, study, model_path, physical, free,
                                 cache_run=getattr(args, "delay_cache_run", None)))
            if cache:
                rows, source = cache
            elif prerequisite:
                rows = [{"fit_id": f"{cell}_seed{seed}_{start_id}", "cell": cell, "seed": seed,
                         "start_id": start_id, "status": "not_run", "reason": prerequisite,
                         "optimization_invoked": False, "optimization_returned": False,
                         "optimizer_success": None, "optimized_vector": None, "final_objective": None,
                         "free_parameters": list(free)} for start_id in matrix["start_ids"]]
            else:
                for start_id in matrix["start_ids"]:
                    initialization = config_to_dict(study.starts[start_id])
                    start = {name: initialization[name] for name in free}
                    if cell == "C_delay_only":
                        start = {"delay_s": study.optimizer["delay_only_start_s"]}
                    fit_id = f"{cell}_seed{seed}_{start_id}"
                    rows.append(fit_and_retain(training, physical, free, start, fit_id, cell, start_id,
                                        source / fit_id, {**provenance, "seed": seed, "cell": cell,
                                            "dataset_input": datasets[f"{scenario}_seed{seed}"]}))
            retain([{**row, "reused": bool(cache)} for row in rows])
            progress.update(phase="training_selection", active_fit=None)
            checkpoint()
            selection = select_rows(rows, free)
            selection_path = directory / "selection.json"
            write_json(selection_path, selection)
            selection_hash = sha256_file(selection_path)
            # Only sealed candidates can enter truth/held-out scoring.
            progress["phase"] = "evaluation"
            checkpoint()
            nominal = evaluate_model(data, study.nominal, model_path, study, delay_s=0.)
            oracle = evaluate_model(data, truth.model, model_path, study, delay_s=data.delay_s)
            evaluated_rows = []
            for row in rows:
                artifact = source / row["result"]["path"] if cache and cell == "B" else source / row["fit_id"] / "result.json"
                retained = {**row, "reused": bool(cache), "source_result": file_ref(artifact, root) if artifact.is_file() else None}
                retained["parameter_rows"] = enrich_parameters(row, {"mass_kg": truth.model.mass_kg,
                    "damping_nm_s_per_rad": truth.model.damping_nm_s_per_rad, "delay_s": data.delay_s})
                if row.get("optimization_returned"):
                    values = dict(zip(free, row["optimized_vector"], strict=True))
                    fitted = replace(physical, **{name: value for name, value in values.items() if name != "delay_s"})
                    delay = values.get("delay_s", 0.)
                    prediction = evaluate_model(data, fitted, model_path, study, delay_s=delay)
                    retained.update(fitted_physics=config_to_dict(fitted), fitted_delay_s=delay,
                                    metrics=prediction, improvement_relative_to_nominal={split: relative_improvement(prediction[split], nominal[split]) for split in ("train", "heldout")})
                    # Unequal weak/rich horizons are never equated by raw loss.
                    shared_cost = sum(item["n_samples"] * item["normalized_rmse"] ** 2 for item in prediction["train"]["trajectory_metrics"])
                    if not np.isclose(shared_cost, row["final_objective"], rtol=1e-8, atol=1e-7):
                        raise ValueError(f"Shared evaluator disagrees with official objective: {row['fit_id']}")
                    retained["objective_equivalence"] = {"passed": True, "shared_cost": shared_cost}
                evaluated_rows.append(retained)
            chosen = next((row for row in evaluated_rows if row["fit_id"] == selection["selected_fit_id"]), None)
            group = {"group_id": group_id, "cell": cell, "seed": seed, "scenario": scenario,
                     "selection": file_ref(selection_path, root), "selected_fit_id": selection["selected_fit_id"],
                     "nominal": nominal, "oracle": oracle, "fits": [row["fit_id"] for row in rows],
                     "oracle_definition": "Ground-truth physics and target delay in fitting family; omitted force absent.",
                     "generator_aware_noise_floor": oracle["generator_aware_noise_floor"],
                     "selection_unchanged": sha256_file(selection_path) == selection_hash}
            if cell == "B" and chosen:
                values = dict(zip(free, chosen["optimized_vector"], strict=True))
                eligible = all(abs(values[name] - getattr(truth.model, name)) / abs(getattr(truth.model, name))
                               <= study.criteria["delay_physical_eligibility_relative_error_max"] for name in free)
                selected_B[seed] = {"physical": replace(study.nominal, **dict(zip(free, chosen["optimized_vector"], strict=True))),
                                    "physical_eligible": eligible, "fit_id": chosen["fit_id"]}
                group["physical_eligible_for_delay"] = eligible
            if cell == "C_delay_only":
                group["fixed_physics_source_fit_id"] = selected_B.get(seed, {}).get("fit_id")
                group["joint_delay_eligible"] = bool(timing_contract and chosen and abs(chosen["fitted_delay_s"] - data.delay_s) <= study.criteria["delay_absolute_error_max_s"])
                if not prerequisite:
                    group["fixed_physics_zero_delay_metrics"] = evaluate_model(data, physical, model_path, study, delay_s=0.)
                    group["fixed_physics_zero_delay_phase_lag"] = phase_lag_diagnostics(data, physical, model_path, study, delay_s=0.)
            if chosen:
                group["selected_metrics"] = chosen["metrics"]
                group["selected_improvement"] = chosen["improvement_relative_to_nominal"]
                group["selected_parameters"] = chosen["parameter_rows"]
                group["phase_lag"] = phase_lag_diagnostics(data, replace(study.nominal, **{name: value for name, value in
                    zip(free, chosen["optimized_vector"], strict=True) if name != "delay_s"}) if cell != "C_delay_only" else physical,
                    model_path, study, delay_s=chosen["fitted_delay_s"])
            write_json(directory / "evaluation.json", {"group": group, "fits": evaluated_rows,
                                                       "provenance": provenance, "selection_sha256": selection_hash})
            if sha256_file(selection_path) != selection_hash:
                raise ValueError("Post-fit evaluation changed sealed selection")
            groups.append(group)
            retain(evaluated_rows)
            progress.update(phase="group_completed", active_group=None)
            checkpoint()
            print(f"{group_id}: {selection['selected_fit_id']} ({len(rows)} retained starts)", flush=True)
        if cell in ("C_joint", "E"):
            milestone = 6 if cell == "C_joint" and through == 6 else (7 if cell == "C_joint" else 8)
            executed_cells = cells[:cells.index(cell) + 1]
            checkpoint_rows = [row for row in all_rows if row["cell"] in executed_cells or row["cell"] in ("A1", "A2")]
            expected_count = sum(len(specifications[name]["start_ids"]) * len(seeds) for name in executed_cells)
            if through >= 7:
                expected_count += 2
            verified = len(checkpoint_rows) == expected_count and all(row["status"] == "completed" for row in checkpoint_rows)
            milestone_status = "verified" if verified else "incomplete"
            milestones.append({"milestone": milestone, "status": milestone_status, "fits_retained": len(checkpoint_rows),
                               "expected_fit_count": expected_count,
                               "scientific_criteria": "Assessed separately; this status describes execution and prerequisite completeness."})
            checkpoint()
            print(f"Milestone {milestone} checkpoint {milestone_status}", flush=True)
    expected_ids = {f"{cell}_seed{seed}_{start_id}" for cell in cells for seed in seeds for start_id in specifications[cell]["start_ids"]}
    if through >= 7:
        expected_ids.update(("A1_seed101_S0", "A2_seed101_S0"))
    accounting = {"expected_count": len(expected_ids), "retained_count": len(all_rows),
                  "unique_retained_count": len({row["fit_id"] for row in all_rows}),
                  "missing_fit_ids": sorted(expected_ids - {row["fit_id"] for row in all_rows}),
                  "unexpected_fit_ids": sorted({row["fit_id"] for row in all_rows} - expected_ids)}
    complete = len(all_rows) == len(expected_ids) and not accounting["missing_fit_ids"] and not accounting["unexpected_fit_ids"] and all(row["status"] == "completed" for row in all_rows)
    result = {"schema_version": "mujoco-sysid-lab.full-run.v1", "status": "passed" if complete else "incomplete",
              "provenance": provenance, "fits": all_rows, "groups": groups, "datasets": datasets,
              "milestones": milestones, "wall_seconds": perf_counter() - started,
              "registered_matrix_total": 82, "matrix_accounting": accounting,
              "progress": {**progress, "phase": "completed"}, "cache_policy": cache_policy,
              "api_contract": file_ref(output / "api_contract.json", root),
              "optimization_calls_this_invocation": sum(bool(row.get("optimization_invoked")) and not row["reused"] for row in all_rows),
              "reused_registered_fits": sum(row["reused"] for row in all_rows), "through_milestone": through}
    write_json(output / "run.json", result)
    print(f"Study {result['status']}: {output.relative_to(root)} in {result['wall_seconds']:.3f}s", flush=True)
    return 0 if result["status"] == "passed" else 1
