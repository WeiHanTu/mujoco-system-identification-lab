"""Curate immutable synthetic study results without fitting or start selection."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from time import perf_counter
from typing import Any

import numpy as np

from mujoco_sysid_lab.config import ModelConfig
from mujoco_sysid_lab.experiment_config import StudyConfig, load_study_config, load_ground_truth
from mujoco_sysid_lab.provenance import collect_provenance, sha256_file, snapshot_sources, write_json


CELLS = ("B", "weak", "delay_excluded", "C_delay_only", "C_joint", "E")
PHYSICAL_CELLS = ("B", "weak", "delay_excluded", "C_joint", "E")
MODEL_COLORS = {"nominal": "#d97706", "identified": "#007f73", "oracle": "#34495e"}
START_COLORS = {"S0": "#007f73", "S1": "#7451a8", "S2": "#d97706"}
CELL_LABELS = {"B": "Matched", "weak": "Weak step", "delay_excluded": "Delay omitted",
               "C_delay_only": "Delay only", "C_joint": "Joint delay", "E": "Mismatch"}


def _stats(values, denominator=5):
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return {"median": float(np.median(finite)) if finite else None,
            "minimum": min(finite) if finite else None, "maximum": max(finite) if finite else None,
            "finite_count": len(finite), "fixed_denominator": denominator,
            "missing_count": denominator-len(finite),
            "unavailable_reason": None if finite else "No finite available results."}


def _criterion(met, complete, **details):
    return {"met": bool(met), "status": ("passed" if met else "failed") if complete else "incomplete",
            "verification_complete": bool(complete), **details}


def _span(values):
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return {"minimum": min(finite) if finite else None, "maximum": max(finite) if finite else None,
            "span": max(finite)-min(finite) if finite else None, "finite_count": len(finite),
            "unavailable_reason": None if finite else "No finite returned values."}


def _initialization_sensitivity(fits, groups, study):
    result = {}
    for cell in study.evidence_matrix:
        result[cell["id"]] = {}
        for seed in cell["noise_seeds"]:
            rows = [row for row in fits if row["cell"] == cell["id"] and row["seed"] == seed]
            group = next((group for group in groups if group["cell"] == cell["id"] and group["seed"] == seed), None)
            nominal_start = next((row for row in rows if row["start_id"] == "S0"), None)
            chosen = next((row for row in rows if group and row["fit_id"] == group.get("selected_fit_id")), None)
            parameter_spans = {}
            for name in cell["free_parameters"]:
                values = []
                for row in rows:
                    parameter = next((p for p in row.get("parameter_rows", []) if p["name"] == name), None)
                    values.append(parameter.get("estimate") if parameter else None)
                parameter_spans[name] = _span(values)
            result[cell["id"]][str(seed)] = {
                "registered_start_count": len(cell["start_ids"]), "retained_start_count": len(rows),
                "optimization_attempt_count": sum(bool(row.get("optimization_invoked")) for row in rows),
                "returned_result_count": sum(bool(row.get("optimization_returned")) for row in rows),
                "converged_start_count": sum(bool(row.get("optimizer_success")) for row in rows),
                "status_counts": {status: sum(row["status"] == status for row in rows) for status in sorted({row["status"] for row in rows})},
                "variation_applicable": len(cell["start_ids"]) > 1,
                "variation_reason": None if len(cell["start_ids"]) > 1 else "Only one initialization is registered for this cell.",
                "parameter_spans": parameter_spans,
                "training_objective_span": _span([row.get("final_objective") for row in rows]),
                "selected_fit_id": group.get("selected_fit_id") if group else None,
                "nominal_start_S0": {"fit_id": nominal_start["fit_id"], "objective": nominal_start.get("final_objective"),
                                     "metrics": nominal_start.get("metrics"), "parameter_rows": nominal_start.get("parameter_rows"),
                                     "optimizer_success": nominal_start.get("optimizer_success")} if nominal_start else None,
                "multi_start_selected": {"fit_id": chosen["fit_id"], "objective": chosen.get("final_objective"),
                                         "metrics": chosen.get("metrics"), "parameter_rows": chosen.get("parameter_rows"),
                                         "optimizer_success": chosen.get("optimizer_success")} if chosen else None,
                "interpretation": "Selection uses exact finite training costs and exact-equality S0/S1/S2 tie order. Tiny floating-point differences or winner identities do not establish materially better predictions.",
            }
    return result


def _provenance_key(provenance, index):
    if not provenance:
        return None
    keep = ("git", "python", "environment", "mujoco_version", "package_versions", "command",
            "model", "lock", "source_sha256", "source_snapshot")
    compact = {key: provenance[key] for key in keep if key in provenance}
    config = provenance.get("config", {})
    compact["config"] = {key: config[key] for key in ("path", "sha256") if key in config}
    identifier = hashlib.sha256(json.dumps(compact, sort_keys=True, allow_nan=False).encode()).hexdigest()[:20]
    index[identifier] = compact
    return identifier


def build_summary(run: dict, study: StudyConfig) -> dict:
    """Keep every fit and assess frozen targets using all five required seeds."""
    fits = run.get("fits", [])
    if len({row["fit_id"] for row in fits}) != len(fits):
        raise ValueError("duplicate fit identities in retained study")
    expected = {f"{cell['id']}_seed{seed}_{start}" for cell in study.evidence_matrix
                for seed in cell["noise_seeds"] for start in cell["start_ids"]}
    actual = {row["fit_id"] for row in fits}
    if actual-expected:
        raise ValueError("study contains unexpected registered fit identities")
    indexed = {row["fit_id"]: row for row in fits}
    groups = run.get("groups", [])
    if len({group["group_id"] for group in groups}) != len(groups):
        raise ValueError("duplicate group identities in retained study")
    selected = {}
    for group in groups:
        if group["cell"] not in CELLS or group["seed"] not in study.noise_seeds:
            raise ValueError("unknown study group identity")
        identifier = group.get("selected_fit_id")
        if identifier is not None:
            if identifier not in indexed or identifier not in group["fits"]:
                raise ValueError("selected fit is absent from its retained group")
            selected[(group["cell"], group["seed"])] = indexed[identifier]
    denominator = study.criteria["fixed_seed_denominator"]
    criteria = {}
    b = [selected.get(("B", seed)) for seed in study.noise_seeds]
    valid_b = [row for row in b if row is not None and row.get("parameter_rows")]
    good_b = sum(all(any(parameter["name"] == name and parameter.get("relative_error") is not None
                        and parameter["relative_error"] <= study.criteria["physical_relative_error_max"]
                        for parameter in row["parameter_rows"])
                     for name in ("mass_kg", "damping_nm_s_per_rad")) for row in valid_b)
    criteria["matched_physical_recovery"] = _criterion(
        good_b >= study.criteria["minimum_recovery_seed_count"], len(valid_b) == denominator,
        recovered_seed_count=good_b, required_seed_count=study.criteria["minimum_recovery_seed_count"],
        fixed_seed_denominator=denominator, relative_error_max=study.criteria["physical_relative_error_max"])
    improvement_stats = {}
    each_trajectory = {}
    for sensor in ("qpos", "qvel"):
        key = sensor+"_rmse"
        values = [row.get("improvement_relative_to_nominal", {}).get("heldout", {}).get(key) if row else None for row in b]
        improvement_stats[sensor] = _stats(values, denominator)
        for name in ("heldout_chirp", "heldout_pulse"):
            values = []
            for seed in study.noise_seeds:
                group = next((group for group in groups if group["cell"] == "B" and group["seed"] == seed), None)
                if group and group.get("selected_metrics"):
                    model = group["selected_metrics"]["trajectory_metrics"][name][key]
                    nominal = group["nominal"]["trajectory_metrics"][name][key]
                    values.append(1-model/nominal if nominal else None)
                else:
                    values.append(None)
            each_trajectory[f"{name}:{sensor}"] = _stats(values, denominator)
    complete_heldout = all(item["finite_count"] == denominator for item in [*improvement_stats.values(), *each_trajectory.values()])
    heldout_met = all(item["median"] is not None and item["median"] >= study.criteria["heldout_median_improvement_min"]
                      for item in improvement_stats.values()) and all(
        item["median"] is not None and item["median"] >= 0 for item in each_trajectory.values())
    criteria["matched_heldout_prediction"] = _criterion(heldout_met, complete_heldout,
        median_improvement_min=study.criteria["heldout_median_improvement_min"],
        split_improvement=improvement_stats, per_trajectory_improvement=each_trajectory,
        require_each_heldout_median_nonnegative=True)
    joint_delay = [selected.get(("C_joint", seed)) for seed in study.noise_seeds]
    delays = []
    for row in joint_delay:
        parameter = next((item for item in row.get("parameter_rows", []) if item["name"] == "delay_s"), None) if row else None
        delays.append(parameter.get("absolute_error") if parameter else None)
    delay_count = sum(value is not None and value <= study.criteria["delay_absolute_error_max_s"] for value in delays)
    criteria["joint_delay_recovery"] = _criterion(delay_count >= study.criteria["minimum_recovery_seed_count"],
        all(value is not None for value in delays), recovered_seed_count=delay_count,
        required_seed_count=study.criteria["minimum_recovery_seed_count"], fixed_seed_denominator=denominator,
        absolute_error_max_s=study.criteria["delay_absolute_error_max_s"], absolute_error_s=_stats(delays, denominator))
    bound_rows = {}
    for cell, names in (("B", ("mass_kg", "damping_nm_s_per_rad")),
                        ("C_joint", ("mass_kg", "damping_nm_s_per_rad", "delay_s"))):
        for name in names:
            parameters = [next((p for p in selected.get((cell, seed), {}).get("parameter_rows", []) if p["name"] == name), None)
                          for seed in study.noise_seeds]
            count = sum(bool(parameter and parameter.get("bound_hit")) for parameter in parameters)
            bound_rows[f"{cell}:{name}"] = {"bound_hit_seed_count": count,
                "consistent_bound_pin": count >= study.metrics["consistent_bound_hit_seed_count"],
                "available_seed_count": sum(parameter is not None for parameter in parameters)}
    criteria["no_consistent_required_bound_pinning"] = _criterion(
        not any(item["consistent_bound_pin"] for item in bound_rows.values()),
        all(item["available_seed_count"] == denominator for item in bound_rows.values()),
        consistent_hit_seed_count=study.metrics["consistent_bound_hit_seed_count"], rows=bound_rows)
    seconds = run.get("wall_seconds")
    runtime_known = isinstance(seconds, (int, float)) and not isinstance(seconds, bool) and np.isfinite(seconds)
    criteria["recorded_cpu_runtime"] = _criterion(runtime_known and seconds <= study.criteria["full_runtime_target_s"], runtime_known,
        measured_wall_seconds=seconds, target_seconds=study.criteria["full_runtime_target_s"],
        optimization_calls_this_invocation=run.get("optimization_calls_this_invocation"),
        reused_registered_fits=run.get("reused_registered_fits"),
        fresh_full_matrix_runtime_verified=run.get("reused_registered_fits") == 0,
        scope="Recorded command wall time; immutable cached fits may be reused. This is not an uncached runtime measurement when reused fits are present.")
    provenance_index = {}
    compact_fits = []
    selected_ids = {group.get("selected_fit_id") for group in groups}
    for row in fits:
        diagnostic = row.get("optimization_summary", row.get("summary", {}))
        compact_fits.append({
            "fit_id": row["fit_id"], "cell": row["cell"], "seed": row["seed"], "start_id": row["start_id"],
            "status": row["status"], "selected": row["fit_id"] in selected_ids,
            "oracle_assisted": row["cell"] in ("A1", "A2"),
            "reused": row.get("reused", False), "failure": row.get("failure"), "not_run_reason": row.get("reason"),
            "free_parameter_order": row.get("free_parameters"), "starting_values": row.get("starting_values"),
            "parameter_rows": [{**parameter, "nominal": 0. if parameter["name"] == "delay_s" else getattr(study.nominal, parameter["name"])}
                               for parameter in row.get("parameter_rows", [])],
            "optimized_vector": row.get("optimized_vector"),
            "objective": diagnostic.get("objective"), "native_final_objective": row.get("final_objective"),
            "optimizer": diagnostic.get("optimizer"), "runtime_seconds": diagnostic.get("runtime_seconds"),
            "sensitivity": diagnostic.get("sensitivity"), "confidence_intervals": diagnostic.get("confidence_intervals"),
            "confidence_interval_interpretation": {
                "unreliable_under_model_mismatch": row["cell"] == "E",
                "reason": ("The fitting model omits the generator force; native local intervals do not support coverage of hidden physical truth. Native interval values/assumptions are preserved."
                           if row["cell"] == "E" else
                           "Physical parameters are fixed to the same-seed B estimate; native delay-only intervals omit uncertainty propagated from that physical fit."
                           if row["cell"] == "C_delay_only" else
                           "Local matched-model approximation only; inspect native rank, bound, and convergence caveats. Not empirical five-seed confidence."),
            },
            "call_counts": diagnostic.get("call_counts", row.get("call_counts")),
            "metrics": row.get("metrics"), "improvement_relative_to_nominal": row.get("improvement_relative_to_nominal"),
            "source_result": row.get("source_result"), "source_fit_id": row.get("source_fit_id", row["fit_id"]),
            "objective_trace": row.get("objective_trace"), "native_result": row.get("native_result"),
            "input_array_sha256": row.get("input_array_sha256"),
            "optimization_provenance_id": _provenance_key(row.get("optimization_provenance", row.get("provenance")), provenance_index),
            "report_provenance_id": _provenance_key(row.get("provenance"), provenance_index),
        })
    descriptive = {}
    for cell in CELLS:
        rows = [selected.get((cell, seed)) for seed in study.noise_seeds]
        descriptive[cell] = {"selected_seed_count": sum(row is not None for row in rows),
            "converged_seed_count": sum(bool(row and row.get("optimizer_success")) for row in rows),
            "failed_or_missing_seed_count": sum(row is None for row in rows),
            "physical_relative_errors": {name: _stats([next((p.get("relative_error") for p in row.get("parameter_rows", []) if p["name"] == name), None) if row else None for row in rows], denominator)
                                         for name in ("mass_kg", "damping_nm_s_per_rad")},
            "heldout_normalized_rmse": _stats([row.get("metrics", {}).get("heldout", {}).get("normalized_rmse") if row else None for row in rows], denominator),
            "training_normalized_rmse": _stats([row.get("metrics", {}).get("train", {}).get("normalized_rmse") if row else None for row in rows], denominator)}
    summary = {"schema_version": "mujoco-sysid-lab.final-summary.v1", "scope": "synthetic dynamics identification only",
        "matrix": {"expected_fit_count": len(expected), "retained_fit_count": len(fits),
                   "missing_fit_ids": sorted(expected-actual), "status_counts": {status: sum(row["status"] == status for row in fits) for status in sorted({row["status"] for row in fits})}},
        "criteria": criteria, "registered_criteria": dict(study.criteria), "fits": compact_fits,
        "groups": groups, "seed_summaries": descriptive, "source_provenance": provenance_index,
        "initialization_sensitivity": _initialization_sensitivity(fits, groups, study),
        "datasets": run.get("datasets", {}), "raw_matrix_accounting": run.get("matrix_accounting"),
        "runtime": {key: run.get(key) for key in ("wall_seconds", "optimization_calls_this_invocation", "reused_registered_fits")},
        "metric_definitions": dict(study.metrics), "fixed_seed_denominator": denominator,
        "oracle_semantics": "Truth physics and true delay inside fitting family; omitted force absent under mismatch.",
        "generator_reference_semantics": "Stored clean observations include generator delay and mismatch; finite-sample noise reference, not a strict lower bound.",
        "claim_boundaries": {"synthetic_only": True, "real_robot": False, "HIL": False, "sim_to_real_transfer": False,
                             "physical_actuator_calibration": False, "production_deployment": False,
                             "arbitrary_robot_identifiability": False},
        "selection_policy": "Finite training objective only, including nonconvergence; exact ties in S0/S1/S2 order.",
        "remote_ci_verified": False}
    json.dumps(summary, allow_nan=False)
    return summary


def write_parameter_csv(path: Path, summary: dict) -> None:
    fields = ("fit_id", "cell", "seed", "start_id", "status", "selected", "reused", "parameter", "unit",
              "initial", "nominal", "estimate", "truth", "absolute_error", "relative_error", "relative_error_reason",
              "lower_bound", "upper_bound", "bound_hit", "success", "nfev", "njev", "callback_events",
              "initial_objective", "final_objective", "runtime_seconds", "source_result", "source_sha256",
              "optimization_provenance_id", "unavailable_reason")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for fit in summary["fits"]:
            parameters = {row["name"]: row for row in fit["parameter_rows"]}
            for name in fit.get("free_parameter_order") or ():
                parameter = parameters.get(name, {})
                row = {key: fit.get(key) for key in ("fit_id", "cell", "seed", "start_id", "status", "selected", "reused", "runtime_seconds", "optimization_provenance_id")}
                row.update({key: parameter.get(key) for key in fields if key in parameter})
                objective, optimizer = fit.get("objective") or {}, fit.get("optimizer") or {}
                row.update(parameter=name, success=optimizer.get("success"), nfev=optimizer.get("nfev"),
                           njev=optimizer.get("njev"), callback_events=optimizer.get("callback_events"),
                           initial_objective=objective.get("initial"), final_objective=fit.get("native_final_objective"),
                           source_result=(fit.get("source_result") or {}).get("path"),
                           source_sha256=(fit.get("source_result") or {}).get("sha256"),
                           unavailable_reason=None if parameter else fit.get("not_run_reason") or (fit.get("failure") or {}).get("message") or "Parameter diagnostic unavailable")
                writer.writerow(row)


def _verify_ref(root: Path, reference: dict) -> Path:
    path = root/reference["path"]
    if not path.resolve().is_relative_to(root) or not path.is_file() or path.is_symlink() or sha256_file(path) != reference["sha256"]:
        raise ValueError(f"Retained artifact hash/path mismatch: {reference.get('path')}")
    return path


def render_figures(root: Path, run: dict, study: StudyConfig, output: Path) -> dict:
    """Render stored outcomes and save exact post-selection plotted predictions."""
    from matplotlib import pyplot as plt
    from mujoco_sysid_lab.advanced_data import load_scenario
    from mujoco_sysid_lab.full_evaluation import predict_model

    indexed = {fit["fit_id"]: fit for fit in run["fits"]}
    group = next(group for group in run["groups"] if group["cell"] == "B" and group["seed"] == 101)
    if group.get("selected_fit_id") not in indexed:
        raise ValueError("Primary seed B winner is unavailable; do not choose a different seed for headline plots")
    chosen = indexed[group["selected_fit_id"]]
    data = load_scenario(_verify_ref(root, run["datasets"]["matched_seed101"]))
    truth = load_ground_truth(study.truth_config_path, study)
    physical = {"nominal": study.nominal, "identified": ModelConfig(**chosen["fitted_physics"]), "oracle": truth.model}
    predictions = {label: predict_model(data, model, root/"model/pendulum.xml", study) for label, model in physical.items()}
    arrays = {}
    for name, record in data.records.items():
        for field in ("timestamps", "controls", "measured_qpos", "measured_qvel"):
            arrays[f"{name}__{field}"] = record.arrays[field]
        for label, mapping in predictions.items():
            for sensor in ("qpos", "qvel"):
                arrays[f"{name}__{label}_{sensor}"] = getattr(mapping[name], sensor)
    np.savez_compressed(output/"plotted_predictions.npz", **arrays)
    paths = {}
    for split, names, filename in (("train", ("train_multisine", "train_prbs"), "train_trajectory_fit.png"),
                                   ("heldout", ("heldout_chirp", "heldout_pulse"), "heldout_trajectory_fit.png")):
        figure, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
        for column, name in enumerate(names):
            record = data.records[name]
            times = record.arrays["timestamps"]
            for row, sensor in enumerate(("qpos", "qvel")):
                axis = axes[row, column]
                axis.plot(times, record.arrays["measured_"+sensor][:, 0], color="#8a939a", lw=.5, alpha=.8, label="Measured (noisy)")
                for label in ("nominal", "oracle", "identified"):
                    axis.plot(times, getattr(predictions[label][name], sensor)[:, 0], color=MODEL_COLORS[label],
                              lw=1.1, ls="--" if label == "oracle" else "-", label=label.capitalize())
                axis.set_title(name.replace("_", " "))
                axis.set_ylabel("qpos (rad)" if sensor == "qpos" else "qvel (rad/s)")
                axis.set_xlabel("Time (s)")
                axis.grid(alpha=.2)
                axis.legend(fontsize=8, loc="upper right")
        figure.suptitle(f"{'Training' if split == 'train' else 'Held-out'} synthetic prediction · B seed 101 · training-selected {chosen['start_id']}")
        figure.savefig(output/filename, dpi=150)
        plt.close(figure)
        paths[filename] = {"path": filename, "sha256": sha256_file(output/filename), "source_fit_id": chosen["fit_id"]}
    figure, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    for column, name in enumerate(("heldout_chirp", "heldout_pulse")):
        record = data.records[name]
        for row, sensor in enumerate(("qpos", "qvel")):
            axis = axes[row, column]
            for label in ("oracle", "identified"):
                residual = record.arrays["measured_"+sensor][:, 0]-getattr(predictions[label][name], sensor)[:, 0]
                axis.plot(record.arrays["timestamps"], residual, color=MODEL_COLORS[label], lw=.55, alpha=.7,
                          label=f"Measured − {label}")
            sigma = study.sensor_sigmas[sensor]
            axis.axhline(2*sigma, ls=":", color="#d97706", lw=.8, label="±2 configured noise σ")
            axis.axhline(-2*sigma, ls=":", color="#d97706", lw=.8)
            axis.set_title(name.replace("_", " "))
            axis.set_ylabel("Residual (rad)" if sensor == "qpos" else "Residual (rad/s)")
            axis.set_xlabel("Time (s)")
            axis.legend(fontsize=8)
            axis.grid(alpha=.2)
    figure.suptitle("Held-out residuals · B seed 101 · noise remains visible")
    figure.savefig(output/"residuals.png", dpi=150)
    plt.close(figure)
    paths["residuals.png"] = {"path": "residuals.png", "sha256": sha256_file(output/"residuals.png"), "source_fit_id": chosen["fit_id"]}
    selected_ids = {group.get("selected_fit_id") for group in run["groups"]}
    figure, axes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
    for axis, name, ylabel in zip(axes, ("mass_kg", "damping_nm_s_per_rad", "delay_s"), ("Mass (kg)", "Damping (N m s/rad)", "Shared delay (s)"), strict=True):
        relevant_cells = PHYSICAL_CELLS if name != "delay_s" else ("C_delay_only", "C_joint")
        for fit in run["fits"]:
            if fit["cell"] not in relevant_cells:
                continue
            parameter = next((item for item in fit.get("parameter_rows", []) if item["name"] == name), None)
            if parameter is None:
                continue
            x = relevant_cells.index(fit["cell"]) + (fit["seed"]-103)*.045 + (int(fit["start_id"][-1])-1)*.018
            is_selected = fit["fit_id"] in selected_ids
            axis.scatter(x, parameter["estimate"], c=START_COLORS[fit["start_id"]], s=48 if is_selected else 16,
                         marker="o", alpha=.95 if is_selected else .4, edgecolors="#202830" if is_selected else "none", linewidths=.5)
        true_value = truth.delay_study_s if name == "delay_s" else getattr(truth.model, name)
        axis.axhline(true_value, color="#202830", ls="--", lw=1, label="Generator truth")
        axis.set_xticks(range(len(relevant_cells)), [CELL_LABELS[cell] for cell in relevant_cells], rotation=25)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=.2)
    for start, color in START_COLORS.items():
        axes[0].scatter([], [], c=color, label=start)
    axes[0].legend(fontsize=8)
    figure.suptitle("All five noise seeds and every retained start · larger outlined points selected by training cost")
    figure.savefig(output/"parameter_recovery.png", dpi=150)
    plt.close(figure)
    paths["parameter_recovery.png"] = {"path": "parameter_recovery.png", "sha256": sha256_file(output/"parameter_recovery.png"), "source_cells": list(CELLS)}
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for index, cell in enumerate(CELLS):
        groups = [group for group in run["groups"] if group["cell"] == cell and group.get("selected_fit_id")]
        for split, offset, color in (("train", -.1, "#7451a8"), ("heldout", .1, "#007f73")):
            values = [group["selected_metrics"][split]["normalized_rmse"] for group in groups]
            if values:
                median, lo, hi = np.median(values), min(values), max(values)
                axes[0].errorbar(index+offset, median, yerr=[[median-lo], [hi-median]], fmt="o", color=color,
                                 capsize=4, label=split if index == 0 else None)
        for name, offset, color in (("mass_kg", -.1, "#7451a8"), ("damping_nm_s_per_rad", .1, "#007f73")):
            values = [100*abs(indexed[group["selected_fit_id"]]["fitted_physics"][name]-getattr(truth.model, name))/getattr(truth.model, name) for group in groups]
            if values:
                median, lo, hi = np.median(values), min(values), max(values)
                axes[1].errorbar(index+offset, median, yerr=[[median-lo], [hi-median]], fmt="o", color=color,
                                 capsize=4, label=("Mass" if name == "mass_kg" else "Damping") if index == 0 else None)
    axes[0].set_yscale("log")
    axes[0].axhline(1, color="#202830", ls=":", label="Noise-scale reference")
    axes[0].set_ylabel("Noise-normalized RMSE (log scale)")
    axes[1].set_ylabel("Absolute relative physical error (%)")
    for axis in axes:
        axis.set_xticks(range(len(CELLS)), [CELL_LABELS[cell] for cell in CELLS], rotation=25)
        axis.legend(fontsize=8)
        axis.grid(alpha=.2)
    figure.suptitle("Ablations · five-seed median with min–max whiskers · delay-only physics fixed to each B winner")
    figure.savefig(output/"ablation_comparison.png", dpi=150)
    plt.close(figure)
    paths["ablation_comparison.png"] = {"path": "ablation_comparison.png", "sha256": sha256_file(output/"ablation_comparison.png"), "source_cells": list(CELLS)}
    return paths


def curate_results(root: Path, run_path: Path, output: Path, command: list[str]) -> dict:
    """Write a new rendering record and authored public summaries/figures."""
    started = perf_counter()
    root, output = Path(root).resolve(), Path(output).resolve()
    run_path = Path(run_path).resolve()
    if run_path.is_dir():
        run_path = run_path/"run.json"
    if not run_path.is_relative_to(root/"artifacts") or not output.is_relative_to(root/"artifacts"):
        raise ValueError("Raw run and rendering output must remain under ignored artifacts/")
    if output.exists():
        raise ValueError("Curation output already exists; preserve earlier evidence")
    run = json.loads(run_path.read_text())
    if run.get("schema_version") != "mujoco-sysid-lab.full-run.v1" or run.get("status") not in ("passed", "incomplete"):
        raise ValueError("Curation requires a completed or explicitly incomplete full study record")
    config_path = root/"configs/evidence_v1.json"
    if sha256_file(config_path) != run["provenance"]["config"]["sha256"]:
        raise ValueError("Curation config differs from immutable study run")
    if sha256_file(root/"model/pendulum.xml") != run["provenance"]["model"]["sha256"]:
        raise ValueError("Curation model differs from immutable study run")
    study = load_study_config(config_path)
    truth_relative = study.truth_config_path.relative_to(root).as_posix()
    if sha256_file(study.truth_config_path) != run["provenance"]["source_sha256"].get(truth_relative):
        raise ValueError("Curation truth configuration differs from immutable study source")
    for group in run["groups"]:
        selected = json.loads(_verify_ref(root, group["selection"]).read_text())
        if (selected.get("physical_truth_evaluated") or selected.get("heldout_evaluated")
                or selected["selected_fit_id"] != group.get("selected_fit_id")):
            raise ValueError("Retained training-only selection identity was changed")
    for fit in run["fits"]:
        if fit.get("source_result"):
            _verify_ref(root, fit["source_result"])
    summary = build_summary(run, study)
    from mujoco_sysid_lab.advanced_data import load_scenario
    excitation = {}
    for scenario in ("matched", "weak"):
        data = load_scenario(_verify_ref(root, run["datasets"][f"{scenario}_seed101"]))
        excitation[scenario] = {name: dict(data.records[name].metadata["excitation_diagnostics"])
                                for name in data.training_ids}
    summary["excitation_diagnostics"] = excitation
    summary["weak_rich_comparison"] = {
        scenario: {"total_training_duration_s": sum(row["duration_s"] for row in diagnostics.values()),
                   "sampled_training_energy_nm2_s": sum(row["sampled_torque_energy_nm2_s"] for row in diagnostics.values()),
                   "reset_count": sum(row["reset_count"] for row in diagnostics.values()),
                   "trajectory_active_rms_nm": {name: row["active_torque_rms_nm"] for name, row in diagnostics.items()}}
        for scenario, diagnostics in excitation.items()
    }
    summary["weak_rich_comparison"]["interpretation"] = "Equal nominal training duration and active RMS; actual sampled energy, zero prefixes, reset count, and transient structure differ. Step recovery can succeed for this known-geometry/known-gain 1-DOF model; this is not arbitrary excitation identifiability."
    provenance = collect_provenance(root, config_path, root/"model/pendulum.xml", command)
    provenance.update(created_at_utc=datetime.now(timezone.utc).isoformat(),
                      operation="Post-selection prediction replay and figure curation; no optimization", seed=101)
    output.mkdir(parents=True, exist_ok=False)
    if provenance["git"]["dirty"]:
        provenance["source_snapshot"] = snapshot_sources(root, output/"source_snapshot")
    write_json(output/"provenance.json", provenance)
    figures = render_figures(root, run, study, output)
    write_parameter_csv(output/"parameter_table.csv", summary)
    summary.update(source_run={"path": run_path.relative_to(root).as_posix(), "sha256": sha256_file(run_path)},
                   producer_provenance={"path": (output/"provenance.json").relative_to(root).as_posix(),
                                        "sha256": sha256_file(output/"provenance.json")},
                   registered_config=run["provenance"]["config"],
                   figures=figures, plotted_predictions={"path": (output/"plotted_predictions.npz").relative_to(root).as_posix(),
                                                       "sha256": sha256_file(output/"plotted_predictions.npz")})
    write_json(output/"summary.json", summary)
    copies = [(output/"summary.json", root/"docs/results/summary.json"),
              (output/"parameter_table.csv", root/"docs/results/parameter_table.csv")]
    copies.extend((output/name, root/"docs/images"/name) for name in figures)
    fit_index = {fit["fit_id"]: fit for fit in run["fits"]}
    for group in run["groups"]:
        if group["seed"] != 101 or not group.get("selected_fit_id"):
            continue
        fit = fit_index[group["selected_fit_id"]]
        directory = root/fit["source_official_directory"] if fit.get("source_official_directory") else _verify_ref(root, fit["source_result"]).parent/"official"
        descriptor = fit.get("official_outputs", {}).get("report")
        if descriptor is None:
            continue
        source = directory/descriptor["path"]
        if sha256_file(source) != descriptor["sha256"]:
            raise ValueError("Selected official report hash changed")
        copies.append((source, root/f"docs/reports/full/{group['cell']}/report.html"))
        if group["cell"] == "B":
            copies.append((source, root/"docs/report.html"))
    public = {}
    for source, target in copies:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if sha256_file(source) != sha256_file(target):
            raise ValueError("Public artifact copy differs from retained rendering")
        public[target.relative_to(root).as_posix()] = {"sha256": sha256_file(target), "source": source.relative_to(root).as_posix()}
    result = {"schema_version": "mujoco-sysid-lab.final-curation.v1", "status": "passed",
              "provenance": provenance, "source_run": summary["source_run"],
              "public_artifacts": public, "no_optimization_performed": True,
              "figures": figures, "plotted_predictions": summary["plotted_predictions"],
              "wall_seconds": perf_counter()-started,
              "runtime_scope": "Post-selection rollouts, summary/figure rendering, hash verification, and public copies; separate from recorded experiment command runtime."}
    write_json(output/"run.json", result)
    return result
