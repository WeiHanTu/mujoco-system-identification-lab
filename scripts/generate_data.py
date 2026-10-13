#!/usr/bin/env python3
"""Generate registered synthetic matched-model data and baseline diagnostics."""

from __future__ import annotations

import argparse
from dataclasses import fields
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from time import perf_counter
import traceback


ROOT = Path(__file__).resolve().parents[1]
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["MPLBACKEND"] = "Agg"
os.environ["MPLCONFIGDIR"] = str(ROOT / "artifacts" / "matplotlib")
os.environ["XDG_CACHE_HOME"] = str(ROOT / "artifacts" / "cache")


def main() -> int:
    started = perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/evidence_v1.json")
    parser.add_argument("--seed", type=int, help="One registered noise seed (default: primary seed)")
    parser.add_argument("--output", help="New ignored output directory; omit for a UTC-timestamped run")
    args = parser.parse_args()

    from mujoco_sysid_lab.data_reporting import evaluate_baselines, generate_plots, write_baseline_table
    from mujoco_sysid_lab.dataset import (
        CORE_TRAJECTORIES, TRAIN_TRAJECTORIES, TrainingSequence, array_sha256,
        generate_dataset, load_evaluation_dataset, load_training_sequences, write_dataset,
    )
    from mujoco_sysid_lab.experiment_config import load_ground_truth, load_study_config
    from mujoco_sysid_lab.provenance import collect_provenance, sha256_file, snapshot_sources, write_json

    config_path = (ROOT / args.config).resolve()
    model_path = ROOT / "model" / "pendulum.xml"
    try:
        study = load_study_config(config_path)
        truth = load_ground_truth(study.truth_config_path, study)
        seed = study.primary_seed if args.seed is None else args.seed
        if seed not in study.noise_seeds:
            raise ValueError(f"seed must be one of {study.noise_seeds}")
        default = "artifacts/milestone2/" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        supplied = Path(args.output or default)
        output = (supplied if supplied.is_absolute() else ROOT / supplied).resolve()
        if not output.is_relative_to(ROOT / "artifacts"):
            raise ValueError("Output must be under ignored artifacts/")
        if output.exists():
            raise ValueError("Output already exists. Omit --output for a fresh run; existing evidence is never overwritten.")
        executable = Path(sys.executable).absolute()
        executable_text = executable.relative_to(ROOT).as_posix() if executable.is_relative_to(ROOT) else str(executable)
        provenance = collect_provenance(ROOT, config_path, model_path,
                                        [executable_text, "scripts/generate_data.py", *sys.argv[1:]])
        if provenance["mujoco_version"] != "3.15.0":
            raise ValueError("Registered MuJoCo is 3.15.0; restore the locked environment before generating data.")
        if provenance["python"]["version"] != (ROOT / ".python-version").read_text().strip():
            raise ValueError("Use the Python version pinned in .python-version for retained evidence.")
    except (ValueError, OSError) as error:
        parser.error(str(error))

    output.mkdir(parents=True, exist_ok=False)
    provenance.update(seed=seed, milestone=2, scenario="matched",
                      created_at_utc=datetime.now(timezone.utc).isoformat(),
                      identification_performed=False,
                      runtime_environment={"MPLBACKEND": "Agg", "MPLCONFIGDIR": "artifacts/matplotlib",
                                           "XDG_CACHE_HOME": "artifacts/cache"})
    if provenance["git"]["dirty"]:
        provenance["source_snapshot"] = snapshot_sources(ROOT, output / "source_snapshot")
    record = {"schema_version": "mujoco-sysid-lab.milestone2.v1", "status": "running",
              "provenance": provenance, "identification_performed": False}
    write_json(output / "run.json", record)
    try:
        registered = {
            "schema_version": "mujoco-sysid-lab.registration.v1",
            "before_first_identification": True,
            "identification_runs": 0,
            "scope": "This generated dataset has not entered identification; the evidence config is frozen at the Milestone 2 checkpoint.",
            "provenance": provenance,
            "evidence_config": json.loads(config_path.read_text()),
            "evidence_config_sha256": sha256_file(config_path),
            "ground_truth_config": json.loads(study.truth_config_path.read_text()),
            "ground_truth_config_sha256": sha256_file(study.truth_config_path),
            "freeze_rule": "Any revision requires a new config version and explanation; no held-out or outcome-driven tuning.",
        }
        write_json(output / "registration.json", registered)
        generation_start = perf_counter()
        generated = generate_dataset(study, truth, model_path, seed)
        repeated = generate_dataset(study, truth, model_path, seed)
        repeat_hashes = {}
        for name in CORE_TRAJECTORIES:
            if generated.records[name].metadata != repeated.records[name].metadata:
                raise RuntimeError(f"Repeated metadata changed for {name}")
            repeat_hashes[name] = {}
            for key, array in generated.records[name].arrays.items():
                digest = array_sha256(array)
                if digest != array_sha256(repeated.records[name].arrays[key]):
                    raise RuntimeError(f"Repeated array changed: {name}/{key}")
                repeat_hashes[name][key] = digest
        if generated.metadata != repeated.metadata:
            raise RuntimeError("Repeated dataset metadata changed")
        generation_seconds = perf_counter() - generation_start
        manifest = write_dataset(output / "dataset", generated, provenance)
        stored = load_evaluation_dataset(manifest)
        for name in CORE_TRAJECTORIES:
            for key, array in stored.records[name].arrays.items():
                if array_sha256(array) != repeat_hashes[name][key]:
                    raise RuntimeError(f"Stored roundtrip changed: {name}/{key}")
        training = load_training_sequences(manifest)
        if tuple(sequence.trajectory_name for sequence in training.sequences) != TRAIN_TRAJECTORIES:
            raise RuntimeError("Training view violated trajectory split")
        training_fields = [field.name for field in fields(TrainingSequence)]
        forbidden_fields = {"truth", "metadata", "clean_qpos", "clean_qvel",
                            "nominal_qpos", "nominal_qvel", "heldout"}
        if forbidden_fields.intersection(training_fields) or hasattr(training, "metadata"):
            raise RuntimeError("Training view exposed evaluation-only fields")
        diagnostics = evaluate_baselines(stored, study)
        plots = generate_plots(output / "plots", stored, study)
        write_baseline_table(output / "baseline_metrics.csv", diagnostics, provenance, seed)
        record.update(
            status="passed", seed=seed,
            dataset_manifest={"path": "dataset/manifest.json", "sha256": sha256_file(manifest)},
            registration={"path": "registration.json", "sha256": sha256_file(output / "registration.json")},
            baseline_table={"path": "baseline_metrics.csv", "sha256": sha256_file(output / "baseline_metrics.csv")},
            trajectory_metadata={name: dict(stored.records[name].metadata) for name in CORE_TRAJECTORIES},
            diagnostics=diagnostics, plots=plots,
            verification={
                "same_seed_numeric_arrays_identical": True,
                "same_seed_metadata_identical": True,
                "npz_roundtrip_identical": True,
                "repeated_array_sha256": repeat_hashes,
                "training_ids": [sequence.trajectory_name for sequence in training.sequences],
                "heldout_ids": list(CORE_TRAJECTORIES[2:]),
                "training_view_fields": training_fields,
                "training_view_exposes_truth_or_latent": False,
                "training_view_loads_heldout_arrays": False,
                "training_read_isolation_basis": "Whitelist reader and regression tests that remove held-out files and spy on loaded NPZ array keys.",
                "generation_and_repeat_seconds": generation_seconds,
            },
        )
    except Exception as error:
        record["status"] = "failed"
        record["failure"] = {"type": type(error).__name__, "message": str(error)}
        (output / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
    record["wall_seconds"] = perf_counter() - started
    write_json(output / "run.json", record)
    print(f"Milestone 2 {record['status']}: {output.relative_to(ROOT)}")
    if record["status"] == "passed":
        print("Four complete trajectories; deterministic replay and split checks passed; no identification performed.")
    else:
        print(record["failure"]["message"], file=sys.stderr)
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
