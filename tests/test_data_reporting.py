"""Reporting preserves model labels and independent trajectory splits."""

from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import pytest

from mujoco_sysid_lab.data_reporting import evaluate_baselines, generate_plots
from mujoco_sysid_lab.dataset import TrajectoryRecord, generate_dataset
from mujoco_sysid_lab.experiment_config import load_ground_truth, load_study_config
from mujoco_sysid_lab.provenance import sha256_file


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def inputs():
    study = load_study_config(ROOT / "configs/evidence_v1.json")
    truth = load_ground_truth(study.truth_config_path, study)
    return study, generate_dataset(study, truth, ROOT / "model/pendulum.xml", 101)


def test_report_labels_only_unfitted_baselines_and_retains_per_trajectory_metrics(inputs):
    study, dataset = inputs
    report = evaluate_baselines(dataset, study)
    assert len(report["trajectory_metrics"]) == 8
    assert {row["model"] for row in report["trajectory_metrics"]} == {"nominal", "oracle"}
    assert all(row["n_samples"] == 3941 for row in report["trajectory_metrics"])
    assert report["identified_model"]["value"] is None
    assert report["primary_success_criteria_evaluated"] is False
    aliases = report["registered_metric_field_map"]
    assert all(aliases.get(name, name) in report["metrics_units"]
               for name in study.metrics["per_trajectory"])
    for split in report["split_metrics"].values():
        for baseline in split.values():
            assert baseline["trajectory_count"] == 2
            assert len(baseline["trajectory_metrics"]) == 2


def test_heldout_measurement_changes_cannot_change_training_statistics(inputs):
    study, dataset = inputs
    before = evaluate_baselines(dataset, study)
    records = dict(dataset.records)
    held = records["heldout_chirp"]
    altered = dict(held.arrays)
    altered["measured_qpos"] = altered["measured_qpos"] + 0.5
    records["heldout_chirp"] = TrajectoryRecord(MappingProxyType(altered), held.metadata)
    after = evaluate_baselines(replace(dataset, records=MappingProxyType(records)), study)
    assert after["split_metrics"]["train"] == before["split_metrics"]["train"]
    assert after["split_metrics"]["heldout"] != before["split_metrics"]["heldout"]


def test_plot_descriptors_resolve_relative_to_run_directory(inputs, tmp_path):
    study, dataset = inputs
    descriptors = generate_plots(tmp_path / "plots", dataset, study)
    assert len(descriptors) == 4
    for descriptor in descriptors.values():
        assert descriptor["path"].startswith("plots/")
        image = tmp_path / descriptor["path"]
        assert image.is_file()
        assert sha256_file(image) == descriptor["sha256"]
