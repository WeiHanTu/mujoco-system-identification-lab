#!/usr/bin/env python3
"""Short CPU pipeline verification, separate from the 82 registered fits."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import MappingProxyType
import traceback


ROOT = Path(__file__).resolve().parents[1]
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["MPLBACKEND"] = "Agg"
os.environ["MPLCONFIGDIR"] = str(ROOT / "artifacts" / "matplotlib")
os.environ["XDG_CACHE_HOME"] = str(ROOT / "artifacts" / "cache")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate smoke JSON field: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"non-finite smoke JSON constant: {value}")


def load_smoke_config(path: Path):
    """Validate exact verification schema and derive copies of frozen inputs."""
    from mujoco_sysid_lab.experiment_config import load_ground_truth, load_study_config
    from mujoco_sysid_lab.provenance import sha256_file

    raw = json.loads(Path(path).read_text(), parse_constant=_reject_constant,
                     object_pairs_hook=_unique_object)
    required = {"schema_version", "scope", "base_evidence_config", "base_evidence_sha256",
                "duration_s", "noise_seed", "start_id", "heldout_pulse_override", "criteria"}
    if not isinstance(raw, dict) or set(raw) != required:
        raise ValueError("smoke config must match the exact versioned schema")
    try:
        json.dumps(raw, allow_nan=False)
    except ValueError as exc:
        raise ValueError("smoke config must contain finite JSON values") from exc
    if raw["schema_version"] != "mujoco-sysid-lab.smoke.v1" or raw["scope"] != "verification_only_not_registered_evidence":
        raise ValueError("smoke scope/schema must explicitly label verification-only execution")
    if raw["base_evidence_config"] != "configs/evidence_v1.json":
        raise ValueError("smoke base must be the frozen repository evidence config")
    base = ROOT / raw["base_evidence_config"]
    if sha256_file(base) != raw["base_evidence_sha256"]:
        raise ValueError("frozen base evidence config hash differs; do not silently revise smoke inputs")
    study = load_study_config(base)
    truth = load_ground_truth(study.truth_config_path, study)
    duration = raw["duration_s"]
    if (type(duration) not in (int, float) or duration <= 2*study.prehistory_s
            or duration <= study.zero_control_prefix_s
            or abs(duration/study.timestep_s-round(duration/study.timestep_s)) > 1e-9):
        raise ValueError("smoke duration must be finite, on the timestep grid, and exceed safe margins")
    if type(raw["noise_seed"]) is not int or raw["noise_seed"] not in study.noise_seeds:
        raise ValueError("smoke noise_seed must be one registered integer seed")
    if raw["start_id"] != "S0":
        raise ValueError("smoke uses exactly one nominal S0 initialization")
    criteria = raw["criteria"]
    expected_criteria = {"physical_relative_error_max": .10,
                         "heldout_qpos_improvement_min_exclusive": 0.,
                         "heldout_qvel_improvement_min_exclusive": 0.,
                         "runtime_target_s": 60.}
    if not isinstance(criteria, dict) or criteria != expected_criteria or any(type(value) not in (float, int) for value in criteria.values()):
        raise ValueError("smoke criteria must preserve the explicit verification thresholds")
    pulses = raw["heldout_pulse_override"]
    if not isinstance(pulses, list) or not pulses:
        raise ValueError("smoke heldout pulse override must be a nonempty schedule")
    end_previous = study.zero_control_prefix_s
    for pulse in pulses:
        if not isinstance(pulse, dict) or set(pulse) != {"start_s", "end_s", "amplitude_nm"}:
            raise ValueError("invalid smoke pulse fields")
        if any(type(value) not in (float, int) for value in pulse.values()):
            raise ValueError("smoke pulse values must be finite numbers")
        start, end = pulse["start_s"], pulse["end_s"]
        if not end_previous <= start < end <= duration or abs(pulse["amplitude_nm"]) > study.nominal.control_limit_nm:
            raise ValueError("smoke pulses must be disjoint and within duration/control bounds")
        if any(abs(value/study.timestep_s-round(value/study.timestep_s)) > 1e-9 for value in (start, end)):
            raise ValueError("smoke pulse times must use the timestep grid")
        end_previous = end
    trajectories = {name: replace(trajectory, duration_s=float(duration)*(2 if name == "weak_step" else 1))
                    for name, trajectory in study.trajectories.items()}
    trajectories["heldout_pulse"] = replace(trajectories["heldout_pulse"],
        settings=MappingProxyType({"pulses": tuple(MappingProxyType(dict(pulse)) for pulse in pulses)}))
    reduced = replace(study, experiment_id="synthetic_pendulum_smoke_v1", duration_s=float(duration),
                      evaluation_window_s=(study.prehistory_s, float(duration)-study.prehistory_s),
                      trajectories=MappingProxyType(trajectories))
    return raw, reduced, truth


def main() -> int:
    started = perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/smoke_v1.json")
    parser.add_argument("--output", help="New ignored output path; omit for a unique timestamp")
    args = parser.parse_args()
    from mujoco_sysid_lab.advanced_data import create_scenario, write_scenario
    from mujoco_sysid_lab.evaluation import aggregate_metrics, evaluate_trajectory, relative_improvement
    from mujoco_sysid_lab.experiment_config import config_to_dict
    from mujoco_sysid_lab.fit_reporting import save_official_fit, summarize_fit
    from mujoco_sysid_lab.identification import build_problem
    from mujoco_sysid_lab.provenance import collect_provenance, sha256_file, snapshot_sources, write_json
    from mujoco_sysid_lab.simulation import compile_model, simulate
    import numpy as np

    config_path = (ROOT / args.config).resolve()
    try:
        config, study, truth = load_smoke_config(config_path)
        supplied = Path(args.output or "artifacts/smoke/"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
        output = (supplied if supplied.is_absolute() else ROOT/supplied).resolve()
        if not output.is_relative_to(ROOT/"artifacts"):
            raise ValueError("Smoke output must remain under ignored artifacts/")
        if output.exists():
            raise ValueError("Output already exists; omit --output for a new run to preserve evidence")
        executable = Path(sys.executable).absolute()
        argv0 = executable.relative_to(ROOT).as_posix() if executable.is_relative_to(ROOT) else str(executable)
        provenance = collect_provenance(ROOT, config_path, ROOT/"model/pendulum.xml",
                                        [argv0, "scripts/run_smoke.py", *sys.argv[1:]])
        if provenance["mujoco_version"] != "3.15.0" or provenance["python"]["version"] != "3.12.5":
            raise ValueError("Smoke requires pinned MuJoCo 3.15.0 and Python 3.12.5")
    except (ValueError, OSError) as error:
        parser.error(str(error))
    output.mkdir(parents=True, exist_ok=False)
    effective_config = config_to_dict(study)
    effective_config["truth_config_path"] = study.truth_config_path.relative_to(ROOT).as_posix()
    provenance.update(seed=config["noise_seed"], scope=config["scope"],
                      created_at_utc=datetime.now(timezone.utc).isoformat(),
                      base_evidence_config={"path": config["base_evidence_config"],
                                            "sha256": config["base_evidence_sha256"],
                                            "content": json.loads((ROOT/config["base_evidence_config"]).read_text())},
                      effective_study_config=effective_config)
    if provenance["git"]["dirty"]:
        provenance["source_snapshot"] = snapshot_sources(ROOT, output/"source_snapshot")
    record = {"schema_version": "mujoco-sysid-lab.smoke-run.v1", "status": "running",
              "scope": config["scope"], "registered_evidence_fits": 0,
              "verification_fit_count": 0, "provenance": provenance}
    write_json(output/"run.json", record)
    try:
        scenario = create_scenario(study, truth, ROOT/"model/pendulum.xml", config["noise_seed"], "matched")
        manifest = write_scenario(output/"dataset", scenario, provenance)
        start = study.starts[config["start_id"]]
        problem = build_problem(scenario.training, ROOT/"model/pendulum.xml", study.nominal, study,
                                ("mass_kg", "damping_nm_s_per_rad"),
                                {"mass_kg": start.mass_kg, "damping_nm_s_per_rad": start.damping_nm_s_per_rad})
        execution = problem.run()
        record["verification_fit_count"] = 1
        estimates = {name: float(execution.optimized_params[name].value[0]) for name in problem.free_parameters}
        # Physical truth enters reporting only after the single fit is complete.
        summary = summarize_fit(problem, execution, {name: getattr(truth.model, name) for name in estimates})
        np.savez_compressed(output/"residuals.npz", initial=execution.initial_residual, final=execution.final_residual)
        official = save_official_fit(output/"official", problem, execution,
                                    title="Reduced CPU smoke verification; not full evidence", fit_scope="joint_physical")
        models = {"nominal": study.nominal, "identified": replace(study.nominal, **estimates), "oracle": truth.model}
        metrics = {}
        for label, physical in models.items():
            rows = {"train": [], "heldout": []}
            model = compile_model(ROOT/"model/pendulum.xml", physical)
            for name, data in scenario.records.items():
                arrays = data.arrays
                predicted = simulate(model, arrays["timestamps"], arrays["controls"])
                split = data.metadata["split"]
                rows[split].append({"trajectory": name, "split": split, **evaluate_trajectory(
                    arrays["timestamps"], arrays["measured_qpos"], arrays["measured_qvel"],
                    predicted.qpos, predicted.qvel, study.sensor_sigmas["qpos"], study.sensor_sigmas["qvel"],
                    window_s=study.evaluation_window_for(name))})
            metrics[label] = {split: aggregate_metrics(values) for split, values in rows.items()}
        improvement = relative_improvement(metrics["identified"]["heldout"], metrics["nominal"]["heldout"])
        checks = {
            "optimizer_converged": bool(execution.optimize_result.success),
            "physical_recovery_within_smoke_threshold": all(row["relative_error"] <= config["criteria"]["physical_relative_error_max"] for row in summary["parameter_rows"]),
            "heldout_qpos_improves": improvement["qpos_rmse"] is not None and improvement["qpos_rmse"] > 0,
            "heldout_qvel_improves": improvement["qvel_rmse"] is not None and improvement["qvel_rmse"] > 0,
        }
        record.update(status="passed" if all(checks.values()) else "failed", checks=checks, fit_summary=summary,
                      metrics=metrics, heldout_improvement=improvement, official_outputs=official,
                      dataset_manifest={"path": "dataset/manifest.json", "sha256": sha256_file(manifest)},
                      claim_boundary="Short verification only; no full five-seed, delay, mismatch, runtime, or real-robot claims.")
    except Exception as error:
        record.update(status="failed", failure={"type": type(error).__name__, "message": str(error)})
        (output/"failure.txt").write_text(traceback.format_exc())
    record["wall_seconds"] = perf_counter()-started
    record.setdefault("checks", {})["runtime_within_smoke_target"] = record["wall_seconds"] <= config["criteria"]["runtime_target_s"]
    if not record["checks"]["runtime_within_smoke_target"]:
        record["status"] = "failed"
    write_json(output/"run.json", record)
    print(json.dumps({"status": record["status"], "scope": record["scope"],
                      "output": output.relative_to(ROOT).as_posix(), "wall_seconds": record["wall_seconds"]}, allow_nan=False))
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
