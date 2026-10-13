"""Compact, referential presentation of retained results; no fitting or replay."""

from __future__ import annotations

from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
from time import perf_counter
from typing import Any

from mujoco_sysid_lab.provenance import collect_provenance, sha256_file, snapshot_sources, write_json


MAX_OVERVIEW_BYTES = 100_000
PARAMETER_COLUMNS = ["estimate", "truth", "absolute_error", "relative_error", "bound_hit"]
SPLIT_METRICS = ("qpos_rmse", "qvel_rmse", "normalized_rmse")
PARAMETER_UNITS = {"mass_kg": "kg", "damping_nm_s_per_rad": "N*m*s/rad", "delay_s": "s"}
BASELINE_FIELDS = (
    "group_id", "cell", "scenario", "seed", "baseline", "name", "unit", "value", "truth",
    "absolute_error", "relative_error", "relative_error_reason", "lower_bound", "upper_bound",
    "within_bounds", "bounds_hit", "lower_bound_hit", "upper_bound_hit", "bound_hit_fraction",
    "fixed", "estimated", "bound_interpretation", "variation_status", "source_summary", "source_sha256",
)


def _reference(value: Any, label: str) -> dict | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a path/hash reference")
    path, digest = value.get("path"), value.get("sha256")
    if (not isinstance(path, str) or not path or Path(path).is_absolute()
            or ".." in Path(path).parts or "\\" in path):
        raise ValueError(f"{label} path must remain repository-relative")
    if (not isinstance(digest, str) or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)):
        raise ValueError(f"{label} requires a SHA256 digest")
    return {"path": path, "sha256": digest}


def _splits(value: Any) -> dict | None:
    """Retain aggregate RMSE only; nested observations and percentiles stay full."""
    if value is None:
        return None
    result = {}
    for split in ("train", "heldout"):
        row = value.get(split)
        result[split] = [row.get(key) for key in SPLIT_METRICS] if row else None
    return result


def baseline_parameter_rows(summary: dict, full_summary_ref: dict) -> list[dict]:
    """Evaluate fixed nominal/oracle values against the stored fitting-family oracle.

    Zero truth makes relative error undefined, including zero-versus-zero delay.
    A fixed baseline at a bound is not evidence that an estimated parameter is
    pinned. Physics under mismatch is compared without claiming the omitted
    force is represented by the fitting-family oracle.
    """
    json.dumps(summary, allow_nan=False)
    source = _reference(full_summary_ref, "baseline full summary")
    if source is None:
        raise ValueError("A baseline full-summary reference is required")
    config = summary["registered_config"]["content"]
    bounds = {"mass_kg": config["mass_bounds_kg"],
              "damping_nm_s_per_rad": config["damping_bounds_nm_s_per_rad"],
              "delay_s": config["delay_bounds_s"]}
    fraction = float(config["metrics"]["bound_hit_fraction"])
    if not math.isfinite(fraction) or not 0 <= fraction < .5:
        raise ValueError("Baseline bound-hit fraction is invalid")
    rows = []
    for group in summary["groups"]:
        oracle = group["oracle"]
        for label in ("nominal", "oracle"):
            baseline = group[label]
            for name, unit in PARAMETER_UNITS.items():
                value = float(baseline["delay_s"] if name == "delay_s" else baseline["physical"][name])
                truth = float(oracle["delay_s"] if name == "delay_s" else oracle["physical"][name])
                lower, upper = float(bounds[name]["lower"]), float(bounds[name]["upper"])
                if not all(math.isfinite(number) for number in (value, truth, lower, upper)) or lower >= upper:
                    raise ValueError("Baseline values and ordered parameter bounds must be finite")
                tolerance = fraction*(upper-lower)
                low_hit, high_hit = abs(value-lower) <= tolerance, abs(value-upper) <= tolerance
                absolute_error = abs(value-truth)
                rows.append({
                    "group_id": group["group_id"], "cell": group["cell"], "scenario": group["scenario"],
                    "seed": group["seed"], "baseline": label, "name": name, "unit": unit,
                    "value": value, "truth": truth, "absolute_error": absolute_error,
                    "relative_error": absolute_error/abs(truth) if truth != 0 else None,
                    "relative_error_reason": None if truth != 0 else "Ground-truth value is zero; relative error is undefined.",
                    "lower_bound": lower, "upper_bound": upper, "within_bounds": lower <= value <= upper,
                    "bounds_hit": low_hit or high_hit, "lower_bound_hit": low_hit, "upper_bound_hit": high_hit,
                    "bound_hit_fraction": fraction, "fixed": True, "estimated": False,
                    "bound_interpretation": "Fixed baseline; a boundary value is not an estimated-parameter pin.",
                    "variation_status": "Fixed values, not optimizer initialization or noise-seed estimates.",
                    "source_summary": source["path"], "source_sha256": source["sha256"],
                })
    json.dumps(rows, allow_nan=False)
    return rows


def write_baseline_parameter_csv(path: Path, rows: list[dict]) -> None:
    """Write all retained baseline groups with explicit undefined-error reasons."""
    json.dumps(rows, allow_nan=False)
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=BASELINE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def build_overview(summary: dict, full_summary_ref: dict) -> dict:
    """Index every registered fit, including missing/failed outcomes, without loss claims.

    Parameter arrays use the declared ``parameter_columns``; all detailed
    residuals, intervals, p95 metrics, and initialization trees remain in the
    hash-referenced full summary. This function does not change either input.
    """
    json.dumps(summary, allow_nan=False)
    if summary.get("schema_version") != "mujoco-sysid-lab.final-summary.v1":
        raise ValueError("Overview requires a versioned final summary")
    source = _reference(full_summary_ref, "full summary")
    if source is None:
        raise ValueError("A full-summary path/hash reference is required")
    config = summary["registered_config"]["content"]
    expected = {f"{cell['id']}_seed{seed}_{start}": (cell, seed, start)
                for cell in config["evidence_matrix"] for seed in cell["noise_seeds"]
                for start in cell["start_ids"]}
    rows = summary["fits"]
    if len({row["fit_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate fit identities in full summary")
    indexed = {row["fit_id"]: row for row in rows}
    if set(indexed)-set(expected):
        raise ValueError("Unexpected fit identities in full summary")
    if summary["matrix"]["retained_fit_count"] != len(rows):
        raise ValueError("Retained fit count disagrees with rows")
    if summary["matrix"]["expected_fit_count"] != len(expected):
        raise ValueError("Registered fit count disagrees with frozen matrix")
    fits, initializations, initialization_ids = [], {}, {}
    for identifier, (cell, seed, start) in expected.items():
        row = indexed.get(identifier)
        if row is None:
            fits.append({"fit_id": identifier, "cell": cell["id"], "seed": seed,
                         "start_id": start, "status": "missing", "selected": False,
                         "reason": "Absent from the retained full summary; not a successful fit.",
                         "source_result": None, "parameters": {}})
            continue
        if (row["cell"], row["seed"], row["start_id"]) != (cell["id"], seed, start):
            raise ValueError("Fit identity fields disagree with the frozen matrix")
        parameters = {parameter["name"]: [parameter.get(key) for key in PARAMETER_COLUMNS]
                      for parameter in row.get("parameter_rows", [])}
        starting_values = row.get("starting_values") or {
            parameter["name"]: parameter.get("initial") for parameter in row.get("parameter_rows", [])}
        initialization_key = json.dumps(starting_values, sort_keys=True, separators=(",", ":"))
        if initialization_key not in initialization_ids:
            initialization_id = f"I{len(initializations)}"
            initialization_ids[initialization_key] = initialization_id
            initializations[initialization_id] = starting_values
        fit = {"fit_id": identifier, "cell": row["cell"], "seed": seed, "start_id": start,
               "status": row["status"], "selected": bool(row.get("selected")),
               "optimizer_success": (row.get("optimizer") or {}).get("success"),
               "reused": bool(row.get("reused")), "oracle_assisted": bool(row.get("oracle_assisted")),
               "initialization_id": initialization_ids[initialization_key],
               "parameters": parameters, "native_final_objective": row.get("native_final_objective"),
               "runtime_seconds": row.get("runtime_seconds"),
               "source_result": _reference(row.get("source_result"), "fit source"),
               "provenance_ids": {"optimization": row.get("optimization_provenance_id"),
                                  "report": row.get("report_provenance_id")}}
        if row.get("source_fit_id") != identifier:
            fit["source_fit_id"] = row.get("source_fit_id")
        if row.get("failure") is not None:
            fit["failure"] = row["failure"]
        if row.get("not_run_reason") is not None:
            fit["reason"] = row["not_run_reason"]
        fits.append(fit)
    group_rows = summary["groups"]
    if len({row["group_id"] for row in group_rows}) != len(group_rows):
        raise ValueError("Duplicate group identities in full summary")
    group_index = {row["group_id"]: row for row in group_rows}
    expected_groups = {f"{cell['id']}_seed{seed}": (cell, seed)
                       for cell in config["evidence_matrix"] if cell["id"] not in ("A1", "A2")
                       for seed in cell["noise_seeds"]}
    if set(group_index)-set(expected_groups):
        raise ValueError("Unexpected group identities in full summary")
    groups, aggregate_metrics, aggregate_ids = [], {}, {}
    for identifier, (cell, seed) in expected_groups.items():
        row = group_index.get(identifier)
        if row is None:
            groups.append({"group_id": identifier, "cell": cell["id"], "seed": seed,
                           "selected_fit_id": None, "metrics": None,
                           "reason": "No retained training-only selection for this registered group."})
            continue
        if (row["cell"], row["seed"]) != (cell["id"], seed):
            raise ValueError("Group identity fields disagree with the frozen matrix")
        chosen = row.get("selected_fit_id")
        if chosen is not None and (chosen not in indexed or chosen not in row["fits"]
                                   or indexed[chosen]["cell"] != cell["id"] or indexed[chosen]["seed"] != seed):
            raise ValueError("Selected fit does not belong to its registered group")
        metrics = {"nominal": _splits(row.get("nominal")),
                   "identified": _splits(row.get("selected_metrics")),
                   "fitting_family_oracle": _splits(row.get("oracle")),
                   "generator_noise_reference": _splits(row.get("generator_aware_noise_floor"))}
        metric_ids = {}
        for label, value in metrics.items():
            if value is None:
                metric_ids[label] = None
                continue
            key = json.dumps(value, sort_keys=True, separators=(",", ":"))
            if key not in aggregate_ids:
                metric_id = f"M{len(aggregate_metrics)}"
                aggregate_ids[key] = metric_id
                aggregate_metrics[metric_id] = value
            metric_ids[label] = aggregate_ids[key]
        groups.append({"group_id": identifier, "cell": cell["id"], "seed": seed,
                       "selected_fit_id": chosen, "selection": _reference(row.get("selection"), "selection"),
                       "metrics": metric_ids})
    overview = {
        "schema_version": "mujoco-sysid-lab.overview.v1", "scope": summary["scope"],
        "full_summary": source, "source_run": _reference(summary.get("source_run"), "raw source"),
        "full_summary_producer": _reference(summary.get("producer_provenance"), "full-summary producer"),
        "registered_config": _reference(summary["registered_config"], "registered config"),
        "matrix": {**summary["matrix"], "overview_fit_count": len(fits),
                   "missing_fit_ids": sorted(set(expected)-set(indexed)),
                   "status_counts": {status: sum(row["status"] == status for row in rows)
                                     for status in sorted({row["status"] for row in rows})},
                   "expected_selected_group_count": len(expected_groups), "retained_group_count": len(group_rows)},
        "execution_complete": (len(rows) == len(expected) and len(group_rows) == len(expected_groups)
                               and all(row["status"] == "completed" for row in rows)
                               and all(row.get("selected_fit_id") for row in group_rows)),
        "parameter_columns": PARAMETER_COLUMNS, "initializations": initializations,
        "split_metric_columns": list(SPLIT_METRICS),
        "parameter_units": PARAMETER_UNITS,
        "metric_units": {"qpos_rmse": "rad", "qvel_rmse": "rad/s", "normalized_rmse": "dimensionless"},
        "metric_definitions": summary["metric_definitions"], "fits": fits, "groups": groups,
        "aggregate_metrics": aggregate_metrics,
        "criteria": summary["criteria"], "registered_criteria": summary["registered_criteria"],
        "seed_summaries": summary["seed_summaries"], "fixed_seed_denominator": summary["fixed_seed_denominator"],
        "runtime": summary["runtime"], "selection_policy": summary["selection_policy"],
        "oracle_semantics": summary["oracle_semantics"],
        "generator_reference_semantics": summary["generator_reference_semantics"],
        "claim_boundaries": summary["claim_boundaries"], "remote_ci_verified": summary["remote_ci_verified"],
        "detail_location": "Source provenance IDs, confidence intervals, objective traces, per-trajectory p95 errors, and initialization sensitivity are resolved in full_summary; unavailable outcomes are never discarded.",
    }
    encode_overview(overview)
    return overview


def encode_overview(overview: dict) -> bytes:
    """Strict compact JSON with a fixed size budget for reviewer navigation."""
    encoded = (json.dumps(overview, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode()
    if len(encoded) > MAX_OVERVIEW_BYTES:
        raise ValueError(f"Overview exceeds {MAX_OVERVIEW_BYTES} bytes; retain details through references")
    return encoded


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON value {value}")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key {key}")
        result[key] = value
    return result


def generate_overview(root: Path, summary_path: Path, output: Path, command: list[str]) -> dict:
    """Copy the new overview and fixed-baseline table with a retained producer record."""
    started = perf_counter()
    if Path(summary_path).is_symlink():
        raise ValueError("Full summary must be a regular repository file")
    root, summary_path, output = Path(root).resolve(), Path(summary_path).resolve(), Path(output).resolve()
    if not summary_path.is_relative_to(root) or not summary_path.is_file():
        raise ValueError("Full summary must be a regular repository file")
    if not output.is_relative_to(root/"artifacts") or output == root/"artifacts":
        raise ValueError("Overview producer output must remain under ignored artifacts/")
    if output.exists():
        raise ValueError("Overview output already exists; preserve earlier producer evidence")
    original = summary_path.read_bytes()
    digest = hashlib.sha256(original).hexdigest()
    summary = json.loads(original, parse_constant=_reject_constant, object_pairs_hook=_unique_object)
    overview = build_overview(summary, {"path": summary_path.relative_to(root).as_posix(), "sha256": digest})
    baseline_rows = baseline_parameter_rows(summary, overview["full_summary"])
    provenance = collect_provenance(root, root/"configs/evidence_v1.json", root/"model/pendulum.xml", command)
    provenance.update(created_at_utc=datetime.now(timezone.utc).isoformat(),
                      operation="Compact presentation of retained summary; no optimization, simulation, or selection",
                      seed=None, noise_seeds=summary["registered_config"]["content"]["noise_seeds"])
    output.mkdir(parents=True, exist_ok=False)
    if provenance["git"]["dirty"]:
        provenance["source_snapshot"] = snapshot_sources(root, output/"source_snapshot")
    write_json(output/"provenance.json", provenance)
    write_baseline_parameter_csv(output/"baseline_parameters.csv", baseline_rows)
    overview["baseline_parameters"] = {
        "path": "docs/results/baseline_parameters.csv", "sha256": sha256_file(output/"baseline_parameters.csv"),
        "row_count": len(baseline_rows), "fixed": True, "estimated": False,
        "interpretation": "Values come from stored nominal/oracle evaluations. The oracle defines parameter truth inside the fitting family; omitted mismatch forces remain absent. Fixed zero-delay bound hits are not estimated-parameter pinning. The CSV lists every retained seed/group; fixed baselines have no optimization-start variation.",
    }
    overview["producer_provenance"] = {"path": (output/"provenance.json").relative_to(root).as_posix(),
                                       "sha256": sha256_file(output/"provenance.json")}
    overview["producer"] = {key: provenance[key] for key in ("git", "python", "mujoco_version", "command")}
    encoded = encode_overview(overview)
    (output/"overview.json").write_bytes(encoded)
    if summary_path.read_bytes() != original:
        raise ValueError("Full summary changed during overview production")
    public = root/"docs/results/overview.json"
    public.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(output/"overview.json", public)
    if sha256_file(public) != sha256_file(output/"overview.json"):
        raise ValueError("Public overview differs from its retained producer")
    public_baselines = root/"docs/results/baseline_parameters.csv"
    shutil.copyfile(output/"baseline_parameters.csv", public_baselines)
    if sha256_file(public_baselines) != overview["baseline_parameters"]["sha256"]:
        raise ValueError("Public baseline table differs from its retained producer")
    record = {"schema_version": "mujoco-sysid-lab.overview-production.v1", "status": "passed",
              "provenance": provenance, "full_summary": overview["full_summary"],
              "overview": {"path": (output/"overview.json").relative_to(root).as_posix(),
                           "sha256": sha256_file(output/"overview.json"), "bytes": len(encoded)},
              "public_artifact": {"path": public.relative_to(root).as_posix(), "sha256": sha256_file(public)},
              "baseline_parameters": {"path": (output/"baseline_parameters.csv").relative_to(root).as_posix(),
                                      "sha256": sha256_file(output/"baseline_parameters.csv"), "row_count": len(baseline_rows)},
              "public_artifacts": {
                  public.relative_to(root).as_posix(): {"sha256": sha256_file(public), "source": (output/"overview.json").relative_to(root).as_posix()},
                  public_baselines.relative_to(root).as_posix(): {"sha256": sha256_file(public_baselines), "source": (output/"baseline_parameters.csv").relative_to(root).as_posix()},
              },
              "registered_fit_count": len(overview["fits"]), "selected_group_count": len(overview["groups"]),
              "no_optimization_performed": True, "no_simulation_performed": True,
              "source_bytes_unchanged": True, "wall_seconds": perf_counter()-started}
    write_json(output/"run.json", record)
    return record
