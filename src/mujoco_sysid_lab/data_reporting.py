"""Matched-data baseline diagnostics, with no estimation or start selection."""

from pathlib import Path
import csv

import numpy as np

from mujoco_sysid_lab.dataset import CORE_TRAJECTORIES, Dataset
from mujoco_sysid_lab.evaluation import aggregate_metrics, evaluate_trajectory
from mujoco_sysid_lab.experiment_config import StudyConfig
from mujoco_sysid_lab.provenance import sha256_file


def evaluate_baselines(dataset: Dataset, study: StudyConfig) -> dict:
    """Use the identical evaluator and window for nominal and matched oracle."""
    rows = []
    noise_diagnostics = {}
    for name in CORE_TRAJECTORIES:
        record = dataset.records[name]
        arrays = record.arrays
        for label, prefix in (("nominal", "nominal"), ("oracle", "clean")):
            metrics = evaluate_trajectory(
                arrays["timestamps"], arrays["measured_qpos"], arrays["measured_qvel"],
                arrays[f"{prefix}_qpos"], arrays[f"{prefix}_qvel"],
                study.sensor_sigmas["qpos"], study.sensor_sigmas["qvel"],
                window_s=study.evaluation_window_for(name),
            )
            rows.append({"trajectory": name, "split": record.metadata["split"],
                         "model": label, **metrics})
        noise_diagnostics[name] = {}
        for sensor in ("qpos", "qvel"):
            noise = arrays[f"measured_{sensor}"] - arrays[f"clean_{sensor}"]
            noise_diagnostics[name][sensor] = {
                "configured_sigma": study.sensor_sigmas[sensor],
                "sample_mean": float(noise.mean()),
                "sample_std_ddof_0": float(noise.std()),
                "sample_rms": float(np.sqrt(np.mean(noise**2))),
                "samples": int(noise.size),
                "scope": "entire trajectory, before the evaluation window mask",
            }
    splits = {
        split: {
            label: aggregate_metrics(row for row in rows
                                     if row["split"] == split and row["model"] == label)
            for label in ("nominal", "oracle")
        }
        for split in ("train", "heldout")
    }
    return {
        "trajectory_metrics": rows,
        "split_metrics": splits,
        "noise_diagnostics": noise_diagnostics,
        "identified_model": {"value": None, "reason": "Identification begins at Milestone 3."},
        "oracle_semantics": "Matched generator-consistent physical parameters; residual contains measurement noise.",
        "metrics_units": {
            "qpos_rmse": "rad", "qvel_rmse": "rad/s", "normalized_rmse": "dimensionless",
            "qpos_p95_absolute_error": "rad", "qvel_p95_absolute_error": "rad/s",
        },
        "registered_metric_field_map": {"noise_normalized_rmse": "normalized_rmse"},
        "primary_success_criteria_evaluated": False,
    }


def write_baseline_table(path: Path, diagnostics: dict, provenance: dict, seed: int) -> None:
    """Keep each trajectory's p95 separate and label baseline/source explicitly."""
    fields = (
        "seed", "source_commit", "source_dirty", "trajectory", "split", "model",
        "qpos_rmse", "qvel_rmse", "normalized_rmse",
        "qpos_p95_absolute_error", "qvel_p95_absolute_error", "n_samples",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in diagnostics["trajectory_metrics"]:
            retained = {key: row[key] for key in fields if key in row}
            retained.update(seed=seed, source_commit=provenance["git"]["commit"],
                            source_dirty=provenance["git"]["dirty"])
            writer.writerow(retained)


def generate_plots(directory: Path, dataset: Dataset, study: StudyConfig) -> dict:
    """Plot stored baselines; descriptor paths are relative to directory.parent."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    directory.mkdir(parents=True, exist_ok=False)
    paths = {}
    titles = {
        "train_multisine": "Training: multisine", "train_prbs": "Training: piecewise constant",
        "heldout_chirp": "Held out: chirp", "heldout_pulse": "Held out: pulses",
    }
    gain = study.nominal.motor_gain_nm_per_unit
    for split, names in (("train", CORE_TRAJECTORIES[:2]), ("heldout", CORE_TRAJECTORIES[2:])):
        figure, axes = plt.subplots(3, 2, figsize=(12, 8), sharex="col", layout="constrained")
        for column, name in enumerate(names):
            arrays = dataset.records[name].arrays
            times = arrays["timestamps"]
            axes[0, column].step(times, arrays["controls"][:, 0] * gain, where="post", color="#546e7a")
            axes[0, column].set_title(titles[name])
            for row, sensor in enumerate(("qpos", "qvel"), start=1):
                axis = axes[row, column]
                axis.plot(times, arrays[f"measured_{sensor}"][:, 0], color="#b0b0b0",
                          linewidth=0.7, alpha=0.8, label="Synthetic measured")
                axis.plot(times, arrays[f"clean_{sensor}"][:, 0], color="#00897b",
                          linewidth=1.3, label="Oracle prediction")
                axis.plot(times, arrays[f"nominal_{sensor}"][:, 0], color="#e67e22",
                          linewidth=1.4, linestyle="--", label="Nominal prediction")
                axis.axvspan(0, study.prehistory_s, color="#f0f0f0", alpha=0.6)
                axis.axvspan(times[-1] - study.prehistory_s, times[-1], color="#f0f0f0", alpha=0.6)
            axes[2, column].set_xlabel("Time (s)")
        for row, label in enumerate(("Torque (N m)", "Position (rad)", "Velocity (rad/s)")):
            for column in range(2):
                axes[row, column].set_ylabel(label)
                axes[row, column].grid(alpha=0.18)
        axes[1, 0].legend(loc="upper left", fontsize=8)
        figure.suptitle(f"{split.capitalize()} baselines · seed {dataset.seed} · no estimation")
        path = directory / f"{split}_baselines.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        paths[f"{split}_baselines"] = path

    figure, axes = plt.subplots(2, 2, figsize=(12, 6), layout="constrained")
    for axis, name in zip(axes.flat, CORE_TRAJECTORIES):
        record = dataset.records[name]
        axis.step(record.arrays["timestamps"], record.arrays["controls"][:, 0] * gain,
                  where="post", color="#1565c0", linewidth=0.9)
        axis.set_title(titles[name])
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Torque (N m)")
        axis.set_ylim(-1.6, 1.6)
        axis.grid(alpha=0.18)
    figure.suptitle("Registered excitation trajectories; motor torque limit ±1.5 N m")
    path = directory / "controls.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    paths["controls"] = path

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    grid = np.linspace(-4, 4, 400)
    for axis, sensor in zip(axes, ("qpos", "qvel")):
        noise = np.concatenate([
            (dataset.records[name].arrays[f"measured_{sensor}"]
             - dataset.records[name].arrays[f"clean_{sensor}"]).ravel()
            / study.sensor_sigmas[sensor]
            for name in CORE_TRAJECTORIES
        ])
        axis.hist(noise, bins=50, density=True, color="#90caf9", alpha=0.8,
                  label="Generated noise / sigma")
        axis.plot(grid, np.exp(-grid**2 / 2) / np.sqrt(2 * np.pi),
                  color="#37474f", label="Registered standard normal")
        axis.set_title("Position noise" if sensor == "qpos" else "Velocity noise")
        axis.set_xlabel("Measurement noise / configured sigma")
        axis.set_ylabel("Density")
        axis.grid(alpha=0.18)
    axes[0].legend(fontsize=8)
    figure.suptitle(f"Measurement-noise diagnostic · seed {dataset.seed} · all four trajectories")
    path = directory / "measurement_noise.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    paths["measurement_noise"] = path
    return {name: {"path": path.relative_to(directory.parent).as_posix(), "sha256": sha256_file(path)}
            for name, path in paths.items()}
