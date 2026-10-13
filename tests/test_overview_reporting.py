"""Presentation tests preserve retained evidence; no simulation or optimization."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from mujoco_sysid_lab.overview_reporting import (
    MAX_OVERVIEW_BYTES, baseline_parameter_rows, build_overview, encode_overview, generate_overview,
    write_baseline_parameter_csv,
)


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT/"docs/results/summary.json"


@pytest.fixture(scope="module")
def summary():
    return json.loads(SUMMARY.read_text())


@pytest.fixture
def reference():
    return {"path": "docs/results/summary.json", "sha256": hashlib.sha256(SUMMARY.read_bytes()).hexdigest()}


def test_compact_overview_preserves_all_fit_identities_parameters_starts_and_sources(summary, reference):
    original = json.dumps(summary, sort_keys=True)
    result = build_overview(summary, reference)
    assert json.dumps(summary, sort_keys=True) == original
    assert len(encode_overview(result)) <= MAX_OVERVIEW_BYTES
    assert len(result["fits"]) == 82 and len(result["groups"]) == 30
    assert result["full_summary"] == reference
    assert result["execution_complete"] is True
    for brief, full in zip(result["fits"], summary["fits"]):
        assert brief["fit_id"] == full["fit_id"]
        assert brief["status"] == full["status"]
        assert brief["selected"] == full["selected"]
        assert brief["source_result"] == full["source_result"]
        assert brief["start_id"] == full["start_id"]
        for parameter in full["parameter_rows"]:
            assert brief["parameters"][parameter["name"]] == [parameter.get(key) for key in result["parameter_columns"]]
            assert result["initializations"][brief["initialization_id"]][parameter["name"]] == parameter["initial"]


def test_selected_aggregate_metrics_remain_exact_without_nested_prediction_trees(summary, reference):
    result = build_overview(summary, reference)
    source_labels = {"nominal": "nominal", "identified": "selected_metrics", "fitting_family_oracle": "oracle",
                     "generator_noise_reference": "generator_aware_noise_floor"}
    for group in result["groups"]:
        full = next(row for row in summary["groups"] if row["group_id"] == group["group_id"])
        for label, source in source_labels.items():
            compact = result["aggregate_metrics"][group["metrics"][label]]
            for split in ("train", "heldout"):
                assert compact[split] == [full[source][split][key] for key in result["split_metric_columns"]]
        assert group["selected_fit_id"] == full["selected_fit_id"]
        assert group["selection"] == full["selection"]
    assert result["criteria"] == summary["criteria"]
    assert result["seed_summaries"] == summary["seed_summaries"]
    assert result["runtime"] == summary["runtime"]


def test_partial_summary_enumerates_missing_fit_and_group_without_fabricating_scores(summary, reference):
    partial = deepcopy(summary)
    selected = partial["groups"][0]["selected_fit_id"]
    partial["fits"] = [row for row in partial["fits"] if row["fit_id"] != selected]
    partial["matrix"]["retained_fit_count"] -= 1
    missing_group = partial["groups"].pop(0)["group_id"]
    result = build_overview(partial, reference)
    assert result["execution_complete"] is False
    assert len(result["fits"]) == 82 and len(result["groups"]) == 30
    absent = next(row for row in result["fits"] if row["fit_id"] == selected)
    assert absent["status"] == "missing" and absent["source_result"] is None and absent["parameters"] == {}
    group = next(row for row in result["groups"] if row["group_id"] == missing_group)
    assert group["selected_fit_id"] is None and group["metrics"] is None
    assert result["matrix"]["missing_fit_ids"] == [selected]
    assert result["matrix"]["status_counts"] == {"completed": 81}


def test_failure_and_nonconvergence_remain_visible(summary, reference):
    changed = deepcopy(summary)
    failed = next(row for row in changed["fits"] if row["fit_id"] == "E_seed103_S2")
    failed.update(status="post_optimization_failed", failure={"type": "Error", "message": "retained failure"})
    failed["optimizer"]["success"] = False
    result = build_overview(changed, reference)
    brief = next(row for row in result["fits"] if row["fit_id"] == failed["fit_id"])
    assert brief["failure"] == failed["failure"] and brief["optimizer_success"] is False
    assert result["execution_complete"] is False
    assert result["matrix"]["status_counts"] == {"completed": 81, "post_optimization_failed": 1}


@pytest.mark.parametrize("change", ["duplicate", "unexpected", "identity", "count"])
def test_invalid_fit_accounting_is_rejected(summary, reference, change):
    invalid = deepcopy(summary)
    if change == "duplicate":
        invalid["fits"].append(invalid["fits"][0])
    elif change == "unexpected":
        invalid["fits"][0]["fit_id"] = "unregistered_seed101_S0"
    elif change == "identity":
        invalid["fits"][0]["seed"] = 102
    else:
        invalid["matrix"]["retained_fit_count"] -= 1
    with pytest.raises(ValueError):
        build_overview(invalid, reference)


def test_selected_fit_must_belong_to_group(summary, reference):
    invalid = deepcopy(summary)
    invalid["groups"][0]["selected_fit_id"] = "E_seed101_S0"
    with pytest.raises(ValueError, match="Selected fit"):
        build_overview(invalid, reference)


def test_group_identity_must_match_frozen_matrix(summary, reference):
    invalid = deepcopy(summary)
    invalid["groups"][0]["seed"] = 102
    with pytest.raises(ValueError, match="Group identity"):
        build_overview(invalid, reference)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_values_are_rejected(summary, reference, bad):
    invalid = deepcopy(summary)
    invalid["fits"][0]["parameter_rows"][0]["estimate"] = bad
    with pytest.raises(ValueError):
        build_overview(invalid, reference)


@pytest.mark.parametrize("reference", [None, {"path": "../summary.json", "sha256": "a"*64},
                                     {"path": "/summary.json", "sha256": "a"*64},
                                     {"path": "docs/results/summary.json", "sha256": "not-a-digest"}])
def test_source_reference_requires_valid_repository_path_and_digest(summary, reference):
    with pytest.raises(ValueError):
        build_overview(summary, reference)


def test_size_budget_is_enforced_without_dropping_evidence():
    with pytest.raises(ValueError, match="exceeds"):
        encode_overview({"oversized": "x"*MAX_OVERVIEW_BYTES})


def test_baseline_parameter_rows_keep_all_180_values_with_known_errors(summary, reference):
    rows = baseline_parameter_rows(summary, reference)
    assert len(rows) == 180
    assert len({(row["group_id"], row["baseline"], row["name"]) for row in rows}) == 180
    matched = [row for row in rows if row["group_id"] == "B_seed101" and row["baseline"] == "nominal"]
    mass = next(row for row in matched if row["name"] == "mass_kg")
    damping = next(row for row in matched if row["name"] == "damping_nm_s_per_rad")
    assert mass["value"] == .8 and mass["truth"] == 1.2
    assert mass["absolute_error"] == pytest.approx(.4)
    assert mass["relative_error"] == pytest.approx(1/3)
    assert damping["absolute_error"] == pytest.approx(.11)
    assert damping["relative_error"] == pytest.approx(11/15)
    assert all(row["absolute_error"] == 0. for row in rows if row["baseline"] == "oracle")
    assert all(row["fixed"] is True and row["estimated"] is False for row in rows)
    assert all(row["source_summary"] == reference["path"] and row["source_sha256"] == reference["sha256"] for row in rows)


def test_delayed_baseline_and_zero_truth_relative_error_are_explicit(summary, reference):
    rows = baseline_parameter_rows(summary, reference)
    delayed = next(row for row in rows if row["group_id"] == "C_joint_seed101"
                   and row["baseline"] == "nominal" and row["name"] == "delay_s")
    assert delayed["value"] == 0. and delayed["truth"] == .024
    assert delayed["absolute_error"] == .024 and delayed["relative_error"] == 1.
    matched = next(row for row in rows if row["group_id"] == "B_seed101"
                   and row["baseline"] == "oracle" and row["name"] == "delay_s")
    assert matched["absolute_error"] == 0. and matched["relative_error"] is None
    assert "undefined" in matched["relative_error_reason"]
    assert matched["bounds_hit"] is matched["lower_bound_hit"] is True
    assert matched["fixed"] is True and matched["estimated"] is False
    assert "not an estimated-parameter pin" in matched["bound_interpretation"]
    json.dumps(rows, allow_nan=False)


def test_fixed_baselines_have_zero_parameter_spread_across_noise_seeds(summary, reference):
    rows = baseline_parameter_rows(summary, reference)
    for cell in {row["cell"] for row in rows}:
        for label in ("nominal", "oracle"):
            for name in ("mass_kg", "damping_nm_s_per_rad", "delay_s"):
                values = [row["value"] for row in rows if row["cell"] == cell
                          and row["baseline"] == label and row["name"] == name]
                assert len(values) == 5 and max(values)-min(values) == 0.


def test_baseline_csv_retains_null_reason_and_source_hash(tmp_path, summary, reference):
    import csv
    rows = baseline_parameter_rows(summary, reference)
    path = tmp_path/"baseline_parameters.csv"
    write_baseline_parameter_csv(path, rows)
    with path.open(newline="") as file:
        written = list(csv.DictReader(file))
    assert len(written) == 180
    zero = next(row for row in written if row["group_id"] == "B_seed101"
                and row["baseline"] == "nominal" and row["name"] == "delay_s")
    assert zero["relative_error"] == "" and "undefined" in zero["relative_error_reason"]
    assert zero["fixed"] == "True" and zero["source_sha256"] == reference["sha256"]


@pytest.mark.parametrize("invalid", ["nonfinite", "reversed_bounds"])
def test_invalid_baseline_parameters_and_bounds_are_rejected(summary, reference, invalid):
    changed = deepcopy(summary)
    if invalid == "nonfinite":
        changed["groups"][0]["nominal"]["physical"]["mass_kg"] = float("nan")
    else:
        changed["registered_config"]["content"]["mass_bounds_kg"] = {"lower": 2., "upper": .4}
    with pytest.raises(ValueError):
        baseline_parameter_rows(changed, reference)


@pytest.fixture
def producer(tmp_path, monkeypatch, summary):
    source = tmp_path/"docs/results/summary.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps(summary))
    provenance = {"git": {"commit": "a"*40, "dirty": True, "tracked_changes": [], "untracked_files": []},
                  "python": {"version": "3.12.5"}, "mujoco_version": "3.15.0",
                  "command": {"argv": ["python", "scripts/summarize_results.py"], "working_directory": "."}}
    monkeypatch.setattr("mujoco_sysid_lab.overview_reporting.collect_provenance", lambda *args: deepcopy(provenance))
    snapshots = []

    def snapshot(root, output):
        snapshots.append(output)
        output.mkdir()
        return {"path": str(output.relative_to(root)), "verification_fixture": True}

    monkeypatch.setattr("mujoco_sysid_lab.overview_reporting.snapshot_sources", snapshot)
    return tmp_path, source, snapshots


def test_producer_writes_only_new_overview_and_preserves_original_public_bytes(producer):
    root, source, snapshots = producer
    sentinels = [source, root/"docs/report.html", root/"docs/results/parameter_table.csv"]
    for path in sentinels[1:]:
        path.write_bytes(b"retained evidence must remain unchanged\n")
    original = {path: path.read_bytes() for path in sentinels}
    output = root/"artifacts/overview/verification"
    result = generate_overview(root, source, output, ["python", "scripts/summarize_results.py"])
    assert result["status"] == "passed" and result["source_bytes_unchanged"] is True
    assert result["no_simulation_performed"] is result["no_optimization_performed"] is True
    assert result["overview"]["bytes"] <= MAX_OVERVIEW_BYTES
    assert result["baseline_parameters"]["row_count"] == 180
    assert set(result["public_artifacts"]) == {"docs/results/overview.json", "docs/results/baseline_parameters.csv"}
    assert snapshots == [output/"source_snapshot"]
    assert (root/"docs/results/overview.json").read_bytes() == (output/"overview.json").read_bytes()
    for path, retained in original.items():
        assert path.read_bytes() == retained
    assert result["full_summary"]["sha256"] == hashlib.sha256(original[source]).hexdigest()
    overview = json.loads((output/"overview.json").read_text())
    assert overview["producer"]["git"]["dirty"] is True
    assert overview["producer_provenance"]["sha256"] == hashlib.sha256((output/"provenance.json").read_bytes()).hexdigest()
    assert overview["baseline_parameters"]["sha256"] == hashlib.sha256((output/"baseline_parameters.csv").read_bytes()).hexdigest()
    assert (root/"docs/results/baseline_parameters.csv").read_bytes() == (output/"baseline_parameters.csv").read_bytes()


def test_existing_producer_output_and_external_output_are_rejected(producer):
    root, source, _ = producer
    output = root/"artifacts/overview/existing"
    output.mkdir(parents=True)
    with pytest.raises(ValueError, match="already exists"):
        generate_overview(root, source, output, ["python"])
    with pytest.raises(ValueError, match="ignored artifacts"):
        generate_overview(root, source, root/"docs/overwrite", ["python"])


@pytest.mark.parametrize("text", ['{"broken":NaN}', '{"broken":1e999}', '{"key":1,"key":2}'])
def test_input_json_rejects_nonfinite_and_duplicate_keys(producer, text):
    root, source, _ = producer
    source.write_text(text)
    with pytest.raises((ValueError, KeyError)):
        generate_overview(root, source, root/"artifacts/overview/invalid", ["python"])


def test_source_symlink_is_rejected(producer):
    root, source, _ = producer
    link = source.parent/"alias.json"
    link.symlink_to(source)
    with pytest.raises(ValueError, match="regular"):
        generate_overview(root, link, root/"artifacts/overview/symlink", ["python"])
