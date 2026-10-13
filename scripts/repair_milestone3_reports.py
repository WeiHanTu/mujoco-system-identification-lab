#!/usr/bin/env python3
"""Regenerate official reports from this project's saved fits, without optimization."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
import shutil
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True, help="Previously generated local Milestone 3 run")
    parser.add_argument("--output", help="New ignored directory; omit for a timestamped report repair")
    args = parser.parse_args()
    source = (ROOT / args.source_run).resolve()
    output = (ROOT / (args.output or ("artifacts/milestone3/" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_reports"))).resolve()
    if not source.is_relative_to(ROOT / "artifacts") or not output.is_relative_to(ROOT / "artifacts"):
        parser.error("Source and output must be authored local runs under ignored artifacts/")
    if output.exists():
        parser.error("Output already exists. Omit --output for a fresh repair; original evidence is preserved.")
    import numpy as np
    from mujoco_sysid_lab.dataset import array_sha256, load_training_sequences
    from mujoco_sysid_lab.experiment_config import load_study_config
    from mujoco_sysid_lab.fit_reporting import save_official_fit, summarize_fit
    from mujoco_sysid_lab.identification import FitExecution, build_problem
    from mujoco_sysid_lab.provenance import collect_provenance, sha256_file, snapshot_sources, write_json

    started = perf_counter()
    original = json.loads((source / "run.json").read_text())
    if original["schema_version"] != "mujoco-sysid-lab.milestone3.v1" or len(original["fits"]) != 4:
        parser.error("Source must contain the four authored Milestone 3 fits")
    if {row["fit_id"] for row in original["fits"]} != {"A1_noiseless_sanity", "A2_noiseless_sanity", "A1_seed101_S0", "A2_seed101_S0"}:
        parser.error("Source fit identities differ from the authorized single-parameter checks")
    config = ROOT / original["provenance"]["config"]["path"]
    model = ROOT / original["provenance"]["model"]["path"]
    study = load_study_config(config)
    for path, expected in ((config, original["provenance"]["config"]["sha256"]),
                           (model, original["provenance"]["model"]["sha256"]),
                           (ROOT / "src/mujoco_sysid_lab/identification.py", original["provenance"]["source_sha256"]["src/mujoco_sysid_lab/identification.py"]),
                           (ROOT / "uv.lock", original["provenance"]["lock"]["sha256"])):
        if sha256_file(path) != expected:
            parser.error("Config, model, lock, or optimization plumbing changed; cached fit cannot be reused")
    for path, expected in original["provenance"]["source_sha256"].items():
        if path.startswith("src/") and path != "src/mujoco_sysid_lab/fit_reporting.py":
            if sha256_file(ROOT / path) != expected:
                parser.error("Scientific source changed; report-only cache reuse is invalid")
    executable = Path(sys.executable).absolute()
    executable_text = executable.relative_to(ROOT).as_posix() if executable.is_relative_to(ROOT) else str(executable)
    provenance = collect_provenance(ROOT, config, model,
                                    [executable_text, "scripts/repair_milestone3_reports.py", *sys.argv[1:]])
    if provenance["package_versions"] != original["provenance"]["package_versions"]:
        parser.error("Installed package set differs from the original optimization environment")
    provenance.update(seed=101, milestone=3, scenario="matched", created_at_utc=datetime.now(timezone.utc).isoformat(),
                      truth_config=original["provenance"]["truth_config"],
                      preregistration=original["provenance"]["preregistration"])
    output.mkdir(parents=True, exist_ok=False)
    if provenance["git"]["dirty"]:
        provenance["source_snapshot"] = snapshot_sources(ROOT, output / "source_snapshot")
    record = {"schema_version": "mujoco-sysid-lab.milestone3-report-repair.v1",
              "status": "running", "provenance": provenance,
              "optimization_provenance": original["provenance"],
              "source_run": {"path": source.relative_to(ROOT).as_posix(), "sha256": sha256_file(source / "run.json")},
              "optimization_performed_this_invocation": False, "fits": [],
              "scope": original["scope"], "registered_matrix_total": 82}
    write_json(output / "run.json", record)
    try:
        registration = json.loads((source / "registration.json").read_text())
        if (registration["evidence_config_sha256"] != sha256_file(config)
                or registration["ground_truth_config_sha256"] != sha256_file(study.truth_config_path)):
            raise ValueError("Frozen evidence or truth configuration changed")
        write_json(output / "registration.json", registration)
        dataset_manifest = ROOT / original["dataset_input"]["path"]
        if sha256_file(dataset_manifest) != original["dataset_input"]["sha256"]:
            raise ValueError("Source dataset manifest changed")
        training = load_training_sequences(dataset_manifest)
        for original_row in original["fits"]:
            fit_id = original_row["fit_id"]
            old_fit = source / "fits" / fit_id
            destination = output / "fits" / fit_id
            destination.mkdir(parents=True, exist_ok=False)
            row = dict(original_row)
            row.update(status="running", provenance=provenance,
                       optimization_provenance=original["provenance"],
                       optimization_reused=True, optimization_invoked_this_command=False,
                       original_fit_record_sha256=sha256_file(old_fit / "result.json"))
            if "failure" in row:
                row["original_reporting_failure"] = row.pop("failure")
            write_json(destination / "result.json", row)
            try:
                if sha256_file(old_fit / "residuals.npz") != original_row["residuals_sha256"]:
                    raise ValueError("Cached residual/Jacobian file changed")
                if sha256_file(old_fit / "objective_trace.json") != original_row["objective_trace_sha256"]:
                    raise ValueError("Cached callback trace changed")
                # This pickle was created by the local official save_results in
                # the source run. External or user-supplied pickle inputs are
                # not accepted by this repair command.
                native_path = old_fit / "official/results.pkl"
                if native_path.is_symlink() or not native_path.resolve().is_relative_to(source):
                    raise ValueError("Cached native result must be a regular authored file in the source run")
                row["original_native_result_sha256"] = sha256_file(native_path)
                with native_path.open("rb") as handle:
                    result = pickle.load(handle)
                with np.load(old_fit / "residuals.npz", allow_pickle=False) as arrays:
                    initial = arrays["initial"].copy()
                    final = arrays["final"].copy()
                    np.testing.assert_array_equal(result.fun, final)
                    np.testing.assert_array_equal(result.x, arrays["optimized_vector"])
                    np.testing.assert_array_equal(result.jac, arrays["jacobian"])
                parameter = original_row["free_parameters"][0]
                physical = replace(study.nominal, **original_row["initial_physics"])
                input_names = {"timestamps", "controls", "measured_qpos", "measured_qvel", "initial_state"}
                for sequence in training.sequences:
                    path = old_fit / f"{sequence.trajectory_name}_inputs.npz"
                    with np.load(path, allow_pickle=False) as arrays:
                        if set(arrays.files) != input_names:
                            raise ValueError("Cached training array names changed")
                        for name in arrays.files:
                            if array_sha256(arrays[name]) != original_row["input_array_sha256"][sequence.trajectory_name][name]:
                                raise ValueError("Cached training inputs changed")
                problem = build_problem(training, model, physical, study, (parameter,),
                                        {parameter: original_row["summary"]["parameter_rows"][0]["initial"]})
                if original_row["noiseless_sanity"]:
                    # Rebuild sanitized inputs from the saved noiseless observed
                    # arrays, without reading truth or generating data again.
                    from mujoco_sysid_lab.dataset import TrainingDataset
                    rebuilt = []
                    for sequence in training.sequences:
                        path = old_fit / f"{sequence.trajectory_name}_inputs.npz"
                        with np.load(path, allow_pickle=False) as arrays:
                            for name in arrays.files:
                                if array_sha256(arrays[name]) != original_row["input_array_sha256"][sequence.trajectory_name][name]:
                                    raise ValueError("Cached noiseless training inputs changed")
                            rebuilt.append(replace(sequence, **{name: arrays[name].copy() for name in arrays.files}))
                    problem = build_problem(TrainingDataset(seed=101, sequences=tuple(rebuilt)), model,
                                            physical, study, (parameter,),
                                            {parameter: original_row["summary"]["parameter_rows"][0]["initial"]})
                optimized = problem.params.copy()
                optimized.update_from_vector(result.x)
                np.testing.assert_array_equal(problem.evaluate(problem.params.as_vector()), initial)
                np.testing.assert_array_equal(problem.evaluate(result.x), final)
                trace = json.loads((old_fit / "objective_trace.json").read_text())
                np.testing.assert_array_equal(result.extras["objective"], [item["objective"] for item in trace])
                np.testing.assert_array_equal(result.extras["candidate"], [item["candidate"] for item in trace])
                summary = original_row["summary"]
                execution = FitExecution(optimized, result, initial, final, [], [], [], [],
                                         summary["call_counts"], tuple(trace),
                                         summary["optimizer"]["callback_events"], summary["runtime_seconds"])
                truth_values = {parameter: summary["parameter_rows"][0]["truth"]}
                reproduced = summarize_fit(problem, execution, truth_values,
                                           noiseless=row["noiseless_sanity"],
                                           bound_hit_fraction=study.metrics["bound_hit_fraction"])
                if reproduced != summary:
                    raise ValueError("Cached fit summary changed during report repair")
                row["official_outputs"] = save_official_fit(
                    destination / "official", problem, execution,
                    title=f"{fit_id}: synthetic oracle-assisted single-parameter check",
                    noiseless=row["noiseless_sanity"])
                for name in ("residuals.npz", "objective_trace.json", "optimizer.log",
                             "train_multisine_inputs.npz", "train_prbs_inputs.npz"):
                    shutil.copyfile(old_fit / name, destination / name)
                threshold = study.criteria["noiseless_single_parameter_relative_error_max"] if row["noiseless_sanity"] else study.criteria["physical_relative_error_max"]
                relative_error = summary["parameter_rows"][0]["relative_error"]
                row["recovery_check"] = {"relative_error": relative_error, "threshold": threshold,
                                         "passed": relative_error <= threshold,
                                         "scope": "noiseless implementation sanity" if row["noiseless_sanity"] else "one oracle-assisted seed; not multi-seed recovery"}
                row["cached_arrays_and_summary_reproduced_exactly"] = True
                row["status"] = "completed"
            except Exception as error:
                row["status"] = "failed"
                row["failure"] = {"type": type(error).__name__, "message": str(error)}
                (destination / "failure.txt").write_text(traceback.format_exc())
            write_json(destination / "result.json", row)
            record["fits"].append(row)
            write_json(output / "run.json", record)
        rows = record["fits"]
        record["registered_evidence_fit_count"] = sum(row["registered_evidence_fit"] for row in rows if row["status"] == "completed")
        record["sanity_fit_count"] = sum(row["noiseless_sanity"] for row in rows if row["status"] == "completed")
        record["status"] = "passed" if all(row["status"] == "completed" and row["recovery_check"]["passed"] for row in rows) else "failed"
        record["joint_recovery_and_heldout_improvement_evaluated"] = False
    except Exception as error:
        record["status"] = "failed"
        record["failure"] = {"type": type(error).__name__, "message": str(error)}
        (output / "failure.txt").write_text(traceback.format_exc())
    record["wall_seconds"] = perf_counter() - started
    write_json(output / "run.json", record)
    print(f"Milestone 3 report repair {record['status']}: {output.relative_to(ROOT)}")
    print("No optimization performed; original native results and failed reporting run retained.")
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
