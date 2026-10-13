"""Known report fixtures assess frozen criteria, never invent evidence fits."""

from copy import deepcopy
import csv
import json
from pathlib import Path

import pytest

from mujoco_sysid_lab.experiment_config import load_study_config
from mujoco_sysid_lab.final_reporting import build_summary, write_parameter_csv


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def study():
    return load_study_config(ROOT/"configs/evidence_v1.json")


@pytest.fixture
def report_fixture(study):
    """Artificial numbers for schema/criterion unit tests; no optimizer run."""
    fits, groups = [], []
    for cell in study.evidence_matrix:
        for seed in cell["noise_seeds"]:
            identifiers = []
            for start in cell["start_ids"]:
                identifier = f"{cell['id']}_seed{seed}_{start}"
                identifiers.append(identifier)
                parameters = []
                for name in cell["free_parameters"]:
                    truth = {"mass_kg": 1.2, "damping_nm_s_per_rad": .15, "delay_s": .024}[name]
                    parameters.append({"name": name, "estimate": truth, "truth": truth,
                                       "absolute_error": 0., "relative_error": 0., "bound_hit": False})
                fits.append({"fit_id": identifier, "cell": cell["id"], "seed": seed, "start_id": start,
                             "status": "completed", "free_parameters": list(cell["free_parameters"]),
                             "parameter_rows": parameters, "optimizer_success": True,
                             "final_objective": 10., "optimized_vector": [p["estimate"] for p in parameters],
                             "improvement_relative_to_nominal": {"heldout": {"qpos_rmse": .3, "qvel_rmse": .4}},
                             "optimization_summary": {"confidence_intervals": {"caveats": [], "half_widths": [.001]},
                                                      "optimizer": {"success": True}}})
            if cell["id"] in ("A1", "A2"):
                continue
            trajectory_metrics = {name: {"qpos_rmse": .7, "qvel_rmse": .6}
                                  for name in ("heldout_chirp", "heldout_pulse")}
            nominal = {name: {"qpos_rmse": 1., "qvel_rmse": 1.} for name in trajectory_metrics}
            groups.append({"group_id": f"{cell['id']}_seed{seed}", "cell": cell["id"], "seed": seed,
                           "fits": identifiers, "selected_fit_id": identifiers[0],
                           "selected_metrics": {"trajectory_metrics": trajectory_metrics},
                           "nominal": {"trajectory_metrics": nominal}})
    return {"fits": fits, "groups": groups, "wall_seconds": 57.,
            "optimization_calls_this_invocation": 70, "reused_registered_fits": 12}


def test_complete_report_keeps_all_registered_fits_and_assesses_five_seed_targets(study, report_fixture):
    result = build_summary(report_fixture, study)
    assert result["matrix"]["expected_fit_count"] == result["matrix"]["retained_fit_count"] == 82
    assert result["matrix"]["missing_fit_ids"] == []
    assert all(criterion["met"] for criterion in result["criteria"].values())
    assert result["criteria"]["matched_physical_recovery"]["fixed_seed_denominator"] == 5
    assert result["criteria"]["matched_physical_recovery"]["recovered_seed_count"] == 5
    assert result["criteria"]["recorded_cpu_runtime"]["fresh_full_matrix_runtime_verified"] is False
    assert len(result["fits"]) == 82
    json.dumps(result, allow_nan=False)


def test_mismatch_ci_flags_preserve_raw_numbers_without_claiming_physical_coverage(study, report_fixture):
    result = build_summary(report_fixture, study)
    row = next(row for row in result["fits"] if row["cell"] == "E")
    assert row["confidence_intervals"] == {"caveats": [], "half_widths": [.001]}
    assert row["confidence_interval_interpretation"]["unreliable_under_model_mismatch"] is True
    assert "hidden physical truth" in row["confidence_interval_interpretation"]["reason"]


def test_physical_recovery_requires_both_parameters_in_at_least_four_seeds(study, report_fixture):
    changed = deepcopy(report_fixture)
    for row in changed["fits"]:
        if row["cell"] == "B" and row["seed"] in (101, 102) and row["start_id"] == "S0":
            row["parameter_rows"][1]["relative_error"] = .1001
    result = build_summary(changed, study)
    criterion = result["criteria"]["matched_physical_recovery"]
    assert criterion["recovered_seed_count"] == 3
    assert criterion["status"] == "failed"


def test_missing_seed_remains_fixed_denominator_and_incomplete_not_verified(study, report_fixture):
    changed = deepcopy(report_fixture)
    changed["groups"] = [group for group in changed["groups"] if group["group_id"] != "B_seed105"]
    result = build_summary(changed, study)
    criterion = result["criteria"]["matched_physical_recovery"]
    assert criterion["recovered_seed_count"] == 4 and criterion["met"] is True
    assert criterion["status"] == "incomplete" and criterion["verification_complete"] is False
    assert result["criteria"]["matched_heldout_prediction"]["status"] == "incomplete"


def test_each_heldout_trajectory_median_must_be_nonnegative(study, report_fixture):
    changed = deepcopy(report_fixture)
    for group in changed["groups"]:
        if group["cell"] == "B":
            group["selected_metrics"]["trajectory_metrics"]["heldout_pulse"]["qvel_rmse"] = 1.1
    result = build_summary(changed, study)
    assert result["criteria"]["matched_heldout_prediction"]["status"] == "failed"


def test_delay_and_consistent_bound_targets_use_registered_thresholds(study, report_fixture):
    changed = deepcopy(report_fixture)
    for row in changed["fits"]:
        if row["cell"] == "C_joint" and row["start_id"] == "S0":
            delay = next(p for p in row["parameter_rows"] if p["name"] == "delay_s")
            delay["absolute_error"] = .00201
        if row["cell"] == "B" and row["start_id"] == "S0" and row["seed"] <= 103:
            row["parameter_rows"][0]["bound_hit"] = True
    result = build_summary(changed, study)
    assert result["criteria"]["joint_delay_recovery"]["status"] == "failed"
    bounds = result["criteria"]["no_consistent_required_bound_pinning"]
    assert bounds["status"] == "failed" and bounds["rows"]["B:mass_kg"]["bound_hit_seed_count"] == 3


def test_nonconvergence_and_failures_are_retained_without_best_seed_filter(study, report_fixture):
    changed = deepcopy(report_fixture)
    row = next(row for row in changed["fits"] if row["fit_id"] == "E_seed103_S2")
    row.update(status="post_optimization_failed", failure={"type": "Error", "message": "retained failure"}, optimizer_success=False)
    result = build_summary(changed, study)
    retained = next(row for row in result["fits"] if row["fit_id"] == "E_seed103_S2")
    assert retained["status"] == "post_optimization_failed" and retained["failure"]["message"] == "retained failure"
    assert result["matrix"]["retained_fit_count"] == 82


def test_duplicate_and_unknown_fit_ids_are_rejected(study, report_fixture):
    changed = deepcopy(report_fixture)
    changed["fits"].append(deepcopy(changed["fits"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        build_summary(changed, study)
    changed["fits"][-1]["fit_id"] = "unexpected_seed101_S0"
    with pytest.raises(ValueError, match="unexpected"):
        build_summary(changed, study)


def test_parameter_csv_retains_all_172_free_parameter_slots_and_unavailable_reasons(tmp_path, study, report_fixture):
    changed = deepcopy(report_fixture)
    row = next(row for row in changed["fits"] if row["fit_id"] == "C_joint_seed105_S2")
    row.update(status="not_run", reason="Prerequisite failed", parameter_rows=[])
    summary = build_summary(changed, study)
    path = tmp_path/"parameter_table.csv"
    write_parameter_csv(path, summary)
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == 172
    missing = [row for row in rows if row["fit_id"] == "C_joint_seed105_S2"]
    assert len(missing) == 3 and all(row["unavailable_reason"] == "Prerequisite failed" for row in missing)


def test_initialization_spans_and_s0_comparison_preserve_all_starts(study, report_fixture):
    changed = deepcopy(report_fixture)
    for row in changed["fits"]:
        if row["cell"] == "B" and row["seed"] == 101:
            number = int(row["start_id"][-1])
            row["parameter_rows"][0]["estimate"] += number*.00001
            row["final_objective"] += number*.0001
            row["optimization_invoked"] = row["optimization_returned"] = True
    sensitivity = build_summary(changed, study)["initialization_sensitivity"]
    b = sensitivity["B"]["101"]
    assert b["registered_start_count"] == b["retained_start_count"] == b["converged_start_count"] == 3
    assert b["optimization_attempt_count"] == 3
    assert b["parameter_spans"]["mass_kg"]["span"] == pytest.approx(.00002)
    assert b["training_objective_span"]["span"] == pytest.approx(.0002)
    assert b["nominal_start_S0"]["fit_id"] == "B_seed101_S0"
    assert b["multi_start_selected"]["fit_id"] == "B_seed101_S0"
    assert "exact-equality" in b["interpretation"]
    assert sensitivity["C_delay_only"]["101"]["variation_applicable"] is False
    assert sensitivity["A1"]["101"]["variation_applicable"] is False


def test_initialization_sensitivity_recomputes_known_cached_raw_spans_when_present(study):
    path = ROOT/"artifacts/full/20261013_evidence_v1/run.json"
    if not path.exists():
        pytest.skip("Local retained study is unavailable in this checkout; synthetic contract fixtures remain authoritative for CI")
    run = json.loads(path.read_text())
    summary = build_summary(run, study)
    for cell in ("B", "E"):
        rows = [row for row in run["fits"] if row["cell"] == cell and row["seed"] == 101]
        expected = [next(p["estimate"] for p in row["parameter_rows"] if p["name"] == "mass_kg") for row in rows]
        result = summary["initialization_sensitivity"][cell]["101"]
        assert result["parameter_spans"]["mass_kg"]["span"] == max(expected)-min(expected)
        assert result["nominal_start_S0"]["metrics"]["heldout"] == next(row["metrics"]["heldout"] for row in rows if row["start_id"] == "S0")
