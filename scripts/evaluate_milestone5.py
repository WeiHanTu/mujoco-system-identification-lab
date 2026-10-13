#!/usr/bin/env python3
"""Evaluate an immutable joint fit on complete untouched trajectories; no fitting."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["MPLBACKEND"] = "Agg"
os.environ["MPLCONFIGDIR"] = str(ROOT / "artifacts/matplotlib")


def main() -> int:
    import numpy as np
    from mujoco_sysid_lab.dataset import load_evaluation_dataset, array_sha256
    from mujoco_sysid_lab.evaluation import aggregate_metrics, evaluate_trajectory, relative_improvement
    from mujoco_sysid_lab.experiment_config import load_study_config, load_ground_truth
    from mujoco_sysid_lab.joint_experiment import check_registration
    from mujoco_sysid_lab.provenance import collect_provenance, sha256_file, snapshot_sources, write_json
    from mujoco_sysid_lab.simulation import compile_model, simulate

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/evidence_v1.json")
    parser.add_argument("--fit-run", default="artifacts/milestone4/20261013_seed101")
    parser.add_argument("--dataset", default="artifacts/milestone2/20261012_seed101_verified/dataset")
    parser.add_argument("--output")
    args = parser.parse_args()
    output = (ROOT / (args.output or "artifacts/milestone5/" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))).resolve()
    if not output.is_relative_to(ROOT / "artifacts") or output.exists():
        parser.error("Choose a new output under ignored artifacts/; evidence is never overwritten")
    study = load_study_config(ROOT / args.config)
    registration = check_registration(ROOT / args.config, study.truth_config_path, ROOT / "docs/results/registration_v1.json")
    fit_run = ROOT / args.fit_run
    selection_path = fit_run / "selection.json"
    selection = json.loads(selection_path.read_text())
    fit_record = json.loads((fit_run / "run.json").read_text())
    selection_hash = sha256_file(selection_path)
    if selection_hash != fit_record["selection_sha256"] or selection["heldout_evaluated"] or selection["physical_truth_evaluated"]:
        raise ValueError("Training-only selection is missing or changed")
    selected = selection["selected_result"]
    if sha256_file(fit_run / selected["path"]) != selected["sha256"]:
        raise ValueError("Selected numerical result changed")
    if fit_record["provenance"]["config"]["sha256"] != sha256_file(ROOT / args.config):
        raise ValueError("Fit config differs from frozen evaluation config")
    manifest = ROOT / args.dataset / "manifest.json"
    if sha256_file(manifest) != fit_record["dataset_input"]["sha256"]:
        raise ValueError("Evaluation dataset differs from immutable fitting input")
    dataset = load_evaluation_dataset(manifest)
    truth = load_ground_truth(study.truth_config_path, study)
    identified = replace(study.nominal, **dict(zip(selection["free_parameter_order"], selection["selected_vector"], strict=True)))
    provenance = collect_provenance(ROOT, ROOT / args.config, ROOT / "model/pendulum.xml", [".venv/bin/python", "scripts/evaluate_milestone5.py", *sys.argv[1:]])
    provenance.update(seed=dataset.seed, milestone=5, selected_fit_id=selection["selected_fit_id"])
    output.mkdir(parents=True, exist_ok=False)
    if provenance["git"]["dirty"]:
        provenance["source_snapshot"] = snapshot_sources(ROOT, output / "source_snapshot")
    started = perf_counter()
    metrics, predictions, descriptors = {}, {}, {}
    for label, physics in (("nominal", study.nominal), ("identified", identified), ("oracle", truth.model)):
        model = compile_model(ROOT / "model/pendulum.xml", physics)
        rows = {"train": [], "heldout": []}
        predictions[label] = {}
        for name, record in dataset.records.items():
            arrays = record.arrays
            predicted = simulate(model, arrays["timestamps"], arrays["controls"], initial_qpos=float(arrays["initial_state"][1]), initial_qvel=float(arrays["initial_state"][2]))
            predictions[label][name] = predicted
            split = study.trajectories[name].split
            rows[split].append({"trajectory": name, "split": split, **evaluate_trajectory(arrays["timestamps"], arrays["measured_qpos"], arrays["measured_qvel"], predicted.qpos, predicted.qvel, study.sensor_sigmas["qpos"], study.sensor_sigmas["qvel"], window_s=study.evaluation_window_for(name))})
        metrics[label] = {split: aggregate_metrics(values) for split, values in rows.items()}
    for name, record in dataset.records.items():
        arrays = {key: value for key, value in record.arrays.items() if key in ("timestamps", "controls", "initial_state", "measured_qpos", "measured_qvel", "clean_qpos", "clean_qvel")}
        for label in metrics:
            arrays[f"{label}_qpos"] = predictions[label][name].qpos
            arrays[f"{label}_qvel"] = predictions[label][name].qvel
        path = output / f"{name}.npz"
        np.savez_compressed(path, **arrays)
        descriptors[name] = {"path": path.name, "sha256": sha256_file(path), "array_sha256": {key: array_sha256(value) for key, value in arrays.items()}}
    previous = json.loads((fit_run / "evaluation.json").read_text())
    prior = next(row for row in previous["fits"] if row.get("selected"))["training_metrics"]
    for key in ("qpos_rmse", "qvel_rmse", "normalized_rmse"):
        np.testing.assert_allclose(metrics["identified"]["train"][key], prior[key], rtol=1e-12, atol=1e-12)
    if sha256_file(selection_path) != selection_hash:
        raise ValueError("Evaluation mutated immutable training selection")
    result = {"schema_version": "mujoco-sysid-lab.milestone5.v1", "status": "passed", "provenance": provenance, "registration": registration,
              "selection": {"path": selection_path.relative_to(ROOT).as_posix(), "sha256": selection_hash, "fit_id": selection["selected_fit_id"], "unchanged": True},
              "dataset": {"path": manifest.relative_to(ROOT).as_posix(), "sha256": sha256_file(manifest)}, "metrics": metrics,
              "improvement_relative_to_nominal": {split: relative_improvement(metrics["identified"][split], metrics["nominal"][split]) for split in ("train", "heldout")},
              "predictions": descriptors, "optimization_calls": 0, "wall_seconds": perf_counter() - started,
              "scope": "One seed; five-seed acceptance targets remain unevaluated."}
    write_json(output / "run.json", result)
    print(json.dumps({"status": "passed", "output": output.relative_to(ROOT).as_posix(), "heldout": metrics["identified"]["heldout"], "improvement": result["improvement_relative_to_nominal"]["heldout"]}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
