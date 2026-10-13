#!/usr/bin/env python3
"""Verify the pinned API and one deterministic trajectory, without fitting."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
from time import perf_counter
import traceback


ROOT = Path(__file__).resolve().parents[1]
for variable in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "1"
os.environ["MPLBACKEND"] = "Agg"
os.environ["MPLCONFIGDIR"] = str(ROOT / "artifacts" / "matplotlib")
os.environ["XDG_CACHE_HOME"] = str(ROOT / "artifacts" / "cache")


def _setup_records(root: Path) -> list[dict]:
    """Hash completed setup inputs, never the runner's active console output."""
    from mujoco_sysid_lab.provenance import sha256_file

    names = (
        "commands.json", "uv-lock.log", "uv-sync.log",
        "pytest-milestone1.log", "pytest-milestone1-final.log",
    )
    paths = [root / "artifacts" / "setup" / name for name in names]
    return [
        {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}
        for path in paths if path.is_file()
    ]


def _output_directory(argument: str | None) -> Path:
    default = "artifacts/milestone1/" + datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    supplied = Path(argument or default)
    result = (supplied if supplied.is_absolute() else ROOT / supplied).resolve()
    if not result.is_relative_to(ROOT / "artifacts"):
        raise ValueError("Milestone output must be under the ignored artifacts/ directory")
    if result.exists():
        raise ValueError(
            f"Output directory already exists: {result.relative_to(ROOT)}. "
            "Omit --output to create a new timestamped run directory, "
            "or choose a different --output path. Existing evidence is never overwritten."
        )
    return result


def main() -> int:
    start = perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/milestone1.json")
    parser.add_argument("--output", help="New ignored artifact directory (default: UTC run ID)")
    args = parser.parse_args()
    try:
        output = _output_directory(args.output)
    except ValueError as error:
        parser.error(str(error))

    import numpy as np

    from mujoco_sysid_lab.api_check import check_api
    from mujoco_sysid_lab.config import load_config
    from mujoco_sysid_lab.provenance import (
        collect_provenance, sha256_file, snapshot_sources, write_json,
    )
    from mujoco_sysid_lab.simulation import compile_model, simulate

    config_path = (ROOT / args.config).resolve()
    model_path = ROOT / "model" / "pendulum.xml"
    config = load_config(config_path)
    executable = Path(sys.executable).absolute()
    executable_text = (
        executable.relative_to(ROOT).as_posix()
        if executable.is_relative_to(ROOT) else str(executable)
    )
    record = collect_provenance(
        ROOT, config_path, model_path,
        [executable_text, "scripts/verify_milestone1.py", *sys.argv[1:]],
    )
    output.mkdir(parents=True, exist_ok=False)
    record.update({
        "milestone": 1,
        "experiment_id": config.experiment_id,
        "seed": config.seed,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Deterministic synthetic simulation and installed API verification only",
        "identification_performed": False,
        "measurement_noise": {"enabled": False},
        "randomness_used": False,
        "runtime_environment": {
            "MPLBACKEND": "Agg", "MPLCONFIGDIR": "artifacts/matplotlib",
            "XDG_CACHE_HOME": "artifacts/cache",
        },
    })
    record["setup_records"] = _setup_records(ROOT)
    if record["git"]["dirty"]:
        record["source_snapshot"] = snapshot_sources(ROOT, output / "source_snapshot")
    write_json(output / "run.json", record)
    try:
        record["api"] = check_api(model_path, config.ground_truth)
        model = compile_model(model_path, config.ground_truth)
        steps = round(config.duration_s / config.ground_truth.timestep_s)
        times = np.arange(steps + 1, dtype=np.float64) * model.opt.timestep
        controls = np.zeros((times.size, 1), dtype=np.float64)
        gain = config.ground_truth.motor_gain_nm_per_unit
        controls[(times >= 0.10) & (times < 0.30), 0] = 0.70 / gain
        controls[(times >= 0.40) & (times < 0.55), 0] = -0.40 / gain
        record["excitation"] = {
            "kind": "verification_pulses",
            "zero_elsewhere": True,
            "pulses": [
                {"start_s": 0.10, "end_exclusive_s": 0.30, "torque_nm": 0.70},
                {"start_s": 0.40, "end_exclusive_s": 0.55, "torque_nm": -0.40},
            ],
            "terminal_control_unused": True,
            "control_to_torque_gain": gain,
        }
        durations = []
        trajectories = []
        for _ in range(2):
            simulation_start = perf_counter()
            trajectories.append(simulate(
                model, times, controls,
                initial_qpos=config.ground_truth.initial_qpos_rad,
                initial_qvel=config.ground_truth.initial_qvel_rad_s,
            ))
            durations.append(perf_counter() - simulation_start)
        first, second = trajectories
        names = ("timestamps", "controls", "qpos", "qvel", "sensordata")
        equality = {
            name: bool(np.array_equal(getattr(first, name), getattr(second, name)))
            for name in names
        }
        if not all(equality.values()):
            raise RuntimeError("Repeated simulation was not numerically identical")
        if not np.array_equal(first.sensordata, np.column_stack((first.qpos, first.qvel))):
            raise RuntimeError("Sensor/state alignment failed")
        np.savez(output / "trajectory.npz", **{
            name: getattr(first, name) for name in names
        })
        record["simulation"] = {
            "samples": int(times.size),
            "control_intervals": int(steps),
            "duration_s": config.duration_s,
            "timestep_s": float(model.opt.timestep),
            "numeric_repeat_equality": equality,
            "finite": all(bool(np.isfinite(getattr(first, name)).all()) for name in names),
            "qpos_range_rad": [float(first.qpos.min()), float(first.qpos.max())],
            "qvel_range_rad_s": [float(first.qvel.min()), float(first.qvel.max())],
            "max_sensor_state_error": float(np.max(np.abs(
                first.sensordata - np.column_stack((first.qpos, first.qvel))
            ))),
            "simulation_seconds": durations,
            "data": {"path": "trajectory.npz", "sha256": sha256_file(output / "trajectory.npz")},
        }
        from matplotlib import pyplot as plt

        figure, axes = plt.subplots(3, 1, figsize=(9, 6), sharex=True, layout="constrained")
        axes[0].step(times, controls[:, 0] * gain, where="post", color="#546e7a")
        axes[1].plot(times, first.qpos[:, 0], color="#1565c0")
        axes[2].plot(times, first.qvel[:, 0], color="#00897b")
        for axis, label in zip(axes, ("Torque (N m)", "Position (rad)", "Velocity (rad/s)")):
            axis.set_ylabel(label)
            axis.grid(alpha=0.20)
        axes[2].set_xlabel("Time (s)")
        figure.suptitle("Milestone 1: deterministic synthetic pendulum simulation")
        figure.savefig(output / "simulation.png", dpi=160)
        plt.close(figure)
        record["simulation"]["plot"] = {
            "path": "simulation.png", "sha256": sha256_file(output / "simulation.png"),
        }
        record["status"] = "passed"
    except Exception as error:
        record["status"] = "failed"
        record["failure"] = {"type": type(error).__name__, "message": str(error)}
        (output / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
    record["verification_seconds"] = perf_counter() - start
    write_json(output / "run.json", record)
    print(f"Milestone 1 {record['status']}: {output.relative_to(ROOT)}")
    if record["status"] == "passed":
        print(f"{record['simulation']['samples']} samples; repeat equality: exact; no identification performed")
    else:
        print(record["failure"]["message"], file=sys.stderr)
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
