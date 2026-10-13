from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
from mujoco import sysid
import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from mujoco_sysid_lab.dataset import TrainingDataset, TrainingSequence
from mujoco_sysid_lab.experiment_config import load_ground_truth, load_study_config
from mujoco_sysid_lab.fit_reporting import (
    _verify_serialized_model, save_official_fit, summarize_fit, summarize_optimization,
)
from mujoco_sysid_lab.identification import build_problem
from mujoco_sysid_lab.provenance import sha256_file
from mujoco_sysid_lab.simulation import compile_model, simulate


ROOT = Path(__file__).resolve().parents[1]


def diagnostic_fixture(*, estimate=1.2, jacobian=None, noiseless=False):
    params = sysid.ParameterDict()
    params.add(sysid.Parameter("mass_kg", 0.8, 0.4, 2.0))
    fitted = params.copy()
    fitted.update_from_vector(np.array([estimate]))
    residual = np.array([-1.0, 0.0, 1.0, 2.0])
    jacobian = np.array([[1.0], [2.0], [3.0], [4.0]]) if jacobian is None else jacobian
    result = OptimizeResult(x=np.array([estimate]), jac=jacobian, success=True, status=1,
                            message="gtol", cost=3.0, nfev=4, njev=4, optimality=0.0)
    problem = SimpleNamespace(params=params)
    execution = SimpleNamespace(
        optimized_params=fitted, optimize_result=result,
        initial_residual=residual * 2, final_residual=residual,
        callback_events=3, runtime_seconds=0.1,
        call_counts={"optimization": {"residual_calls": 8}, "total": {"residual_calls": 10}},
        objective_trace=({"event": "initial", "candidate": [0.8], "objective": 12.0},
                         {"event": "final", "candidate": [estimate], "objective": 3.0}),
    )
    return problem, execution


def test_summary_reports_truth_errors_registered_bounds_and_actual_diagnostics():
    problem, execution = diagnostic_fixture(estimate=1.19)
    summary = summarize_fit(problem, execution, {"mass_kg": 1.2})
    row = summary["parameter_rows"][0]
    assert row["initial"] == 0.8 and row["estimate"] == 1.19 and row["truth"] == 1.2
    assert row["absolute_error"] == pytest.approx(0.01)
    assert row["relative_error"] == pytest.approx(0.01/1.2)
    assert not row["bound_hit"] and row["unit"] == "kg"
    assert summary["objective"]["initial"] == 12.0
    assert summary["objective"]["final"] == summary["objective"]["native_final_cost"] == 3.0
    assert summary["optimizer"]["success"] is True
    assert summary["optimizer"]["nfev"] == 4
    assert summary["optimizer"]["nit"] is None
    assert "does not expose nit" in summary["optimizer"]["unavailable_reasons"]["nit"]
    assert summary["optimizer"]["callback_events"] == 3
    assert "gtol" in summary["optimizer"]["termination_meaning"]
    assert summary["sensitivity"]["rank"] == 1
    assert summary["sensitivity"]["condition_number"] == 1
    assert summary["sensitivity"]["singular_values"][0] == pytest.approx(np.sqrt(30)*1.6)
    assert summary["call_counts"] == execution.call_counts
    json.dumps(summary, allow_nan=False)


def test_official_intervals_are_half_widths_with_explicit_local_assumptions():
    problem, execution = diagnostic_fixture()
    summary = summarize_fit(problem, execution, {"mass_kg": 1.2})
    covariance, half_widths = sysid.calculate_intervals([execution.final_residual], execution.optimize_result.jac)
    intervals = summary["confidence_intervals"]
    assert intervals["status"] == "available"
    np.testing.assert_allclose(intervals["half_widths"], half_widths)
    np.testing.assert_allclose(intervals["covariance"], covariance)
    np.testing.assert_allclose(intervals["intervals"], [[1.2-half_widths[0], 1.2+half_widths[0]]])
    assert any("not five-seed" in item for item in intervals["assumptions"])


def test_noiseless_checks_do_not_claim_statistical_confidence():
    problem, execution = diagnostic_fixture()
    summary = summarize_fit(problem, execution, {"mass_kg": 1.2}, noiseless=True)
    assert summary["confidence_intervals"]["half_widths"] is None
    assert "Noiseless numerical" in summary["confidence_intervals"]["reason"]


def test_singular_jacobian_uses_null_condition_and_intervals_in_strict_json():
    problem, execution = diagnostic_fixture(jacobian=np.zeros((4, 1)))
    summary = summarize_fit(problem, execution, {"mass_kg": 1.2})
    assert summary["sensitivity"]["status"] == "rank_deficient"
    assert summary["sensitivity"]["rank"] == 0
    assert summary["sensitivity"]["condition_number"] is None
    assert summary["confidence_intervals"]["status"] == "unavailable"
    json.dumps(summary, allow_nan=False)


@pytest.mark.parametrize("jacobian", [np.zeros((3, 1)), np.full((4, 1), np.nan)])
def test_invalid_jacobian_is_unavailable_and_finite(jacobian):
    problem, execution = diagnostic_fixture(jacobian=jacobian)
    summary = summarize_fit(problem, execution, {"mass_kg": 1.2})
    assert summary["sensitivity"]["status"] == "unavailable"
    assert summary["confidence_intervals"]["status"] == "unavailable"
    json.dumps(summary, allow_nan=False)


def test_missing_backend_fields_are_null_with_reasons():
    problem, execution = diagnostic_fixture()
    execution.optimize_result = OptimizeResult(x=np.array([1.2]))
    summary = summarize_fit(problem, execution, {"mass_kg": 1.2})
    for field in ("success", "status", "message", "nfev", "njev", "nit", "cost", "optimality"):
        assert summary["optimizer"][field] is None
        assert summary["optimizer"]["unavailable_reasons"][field]
    assert summary["sensitivity"]["status"] == "unavailable"


def test_bound_hit_width_rule_and_nonconvergence_are_not_hidden():
    problem, execution = diagnostic_fixture(estimate=0.4+0.0015)
    execution.optimize_result.success = False
    execution.optimize_result.status = 0
    summary = summarize_fit(problem, execution, {"mass_kg": 1.2})
    assert summary["parameter_rows"][0]["lower_bound_hit"] is True
    assert summary["optimizer"]["success"] is False
    assert "budget" in summary["optimizer"]["termination_meaning"]
    assert len(summary["confidence_intervals"]["caveats"]) == 2


def test_summary_copies_count_snapshots_and_trace_without_mutating_execution():
    problem, execution = diagnostic_fixture()
    before = deepcopy(execution.call_counts)
    summary = summarize_fit(problem, execution, {"mass_kg": 1.2})
    summary["call_counts"]["optimization"]["residual_calls"] = 999
    summary["objective_trace"][0]["candidate"][0] = 999
    assert execution.call_counts == before
    assert execution.objective_trace[0]["candidate"] == [0.8]


@pytest.mark.parametrize("change,error", [
    (lambda execution: setattr(execution, "final_residual", np.array([np.nan]*4)), "finite real vector"),
    (lambda execution: setattr(execution, "final_residual", np.zeros(3)), "same fixed sample"),
    (lambda execution: setattr(execution, "runtime_seconds", -1), "nonnegative"),
])
def test_invalid_execution_summary_is_rejected(change, error):
    problem, execution = diagnostic_fixture()
    change(execution)
    with pytest.raises(ValueError, match=error):
        summarize_fit(problem, execution, {"mass_kg": 1.2})


def test_missing_truth_and_invalid_bound_fraction_are_rejected():
    problem, execution = diagnostic_fixture()
    with pytest.raises(ValueError, match="missing evaluation truth"):
        summarize_fit(problem, execution, {})
    with pytest.raises(ValueError, match="bound_hit_fraction"):
        summarize_fit(problem, execution, {"mass_kg": 1.2}, bound_hit_fraction=-1)


@pytest.fixture(scope="module")
def reduced_single_parameter_fit():
    """Noiseless unit fixture, not one of the registered evidence fits."""
    study = load_study_config(ROOT / "configs/evidence_v1.json")
    truth = load_ground_truth(study.truth_config_path, study)
    model = compile_model(ROOT / "model/pendulum.xml", truth.model)
    times = np.arange(401) * study.timestep_s
    sequences = []
    for identity, name in enumerate(("train_multisine", "train_prbs")):
        controls = (0.6 * np.sin(2*np.pi*(0.7+identity*0.5)*times))[:, None]
        controls[times < study.zero_control_prefix_s] = 0
        clean = simulate(model, times, controls)
        sequences.append(TrainingSequence(name, identity, times, controls, clean.qpos, clean.qvel,
                                          np.array([0.0, 0.0, 0.0])))
    training = TrainingDataset(101, tuple(sequences))
    physical = replace(study.nominal, damping_nm_s_per_rad=truth.model.damping_nm_s_per_rad)
    problem = build_problem(training, ROOT / "model/pendulum.xml", physical, study,
                            ("mass_kg",), {"mass_kg": study.nominal.mass_kg})
    execution = problem.run()
    return problem, execution


def test_actual_official_save_report_and_xml_use_fitted_values_on_copies(tmp_path, reduced_single_parameter_fit):
    problem, execution = reduced_single_parameter_fit
    source_xml = problem.spec.to_xml()
    saved_counts = deepcopy(execution.call_counts)
    output = tmp_path / "official"
    info = save_official_fit(output, problem, execution, title="Noiseless mass unit check", noiseless=True)
    assert info["status"] == "passed" and info["original_spec_unchanged"]
    assert info["videos_generated"] is False
    assert info["reporting_call_counts"]["residual_calls"] == 3
    assert execution.call_counts == saved_counts and problem.spec.to_xml() == source_xml
    assert set(info["files"]) == {"params_x_0.yaml", "params_x_hat.yaml", "results.pkl", "confidence.pkl", "pendulum.xml", "pendulum.mjb", "report.html"}
    for descriptor in info["files"].values():
        assert sha256_file(output / descriptor["path"]) == descriptor["sha256"]
    saved = mujoco.MjModel.from_xml_path(str(output / "pendulum.xml"))
    original = problem.spec.compile()
    assert saved.body_mass[1] == pytest.approx(float(execution.optimized_params["mass_kg"].value[0]), rel=1e-12)
    assert original.body_mass[1] == 0.8
    np.testing.assert_allclose(saved.body_inertia[1] / saved.body_mass[1], original.body_inertia[1] / original.body_mass[1])
    assert saved.dof_damping[0] == 0.15
    html = (output / "report.html").read_text()
    assert "Project interpretation and claim boundaries" in html
    assert "means measured signals" in html and "starting reference, not hidden ground truth" in html
    assert "not statistical confidence" in html
    assert "six significant digits" in html
    assert "Optimization Trace" in html
    assert not list(output.glob("*.mp4"))
    with pytest.raises(ValueError, match="Output already exists"):
        save_official_fit(output, problem, execution)


def test_actual_single_parameter_summary_retains_numerical_recovery_and_trace(reduced_single_parameter_fit):
    problem, execution = reduced_single_parameter_fit
    summary = summarize_fit(problem, execution, {"mass_kg": 1.2}, noiseless=True)
    assert summary["parameter_rows"][0]["relative_error"] <= 0.001
    assert summary["objective"]["final"] < summary["objective"]["initial"]
    assert summary["objective_trace"][0]["event"] == "initial"
    assert summary["objective_trace"][-1]["event"] == "final"
    assert summary["optimizer"]["nit"] is None
    json.dumps(summary, allow_nan=False)


@pytest.mark.parametrize("mass,damping,expected_mass,expected_damping", [
    (0.432198765, 0.00123456789, 0.432199, 0.00123457),
    (0.799999999, 0.04000001234, 0.8, 0.04),
    (1.1998996613593147, 0.15002534185652405, 1.1999, 0.150025),
    (1.876543219, 0.499999999, 1.87654, 0.5),
    (1.234565, 0.1234565, 1.23456, 0.123456),
    (1.234575, 0.1234575, 1.23457, 0.123457),
])
def test_installed_xml_serialization_precision_against_literal_probe_values(tmp_path, mass, damping, expected_mass, expected_damping):
    """Format fixtures verify serialized candidates without any optimizer."""
    study = load_study_config(ROOT / "configs/evidence_v1.json")
    from mujoco_sysid_lab.simulation import load_spec
    spec = load_spec(ROOT / "model/pendulum.xml",
                     replace(study.nominal, mass_kg=mass, damping_nm_s_per_rad=damping))
    full_precision = spec.compile()
    path = tmp_path / "serialized.xml"
    spec.to_file(str(path))
    saved = mujoco.MjModel.from_xml_path(str(path))
    assert saved.body_mass[1] == expected_mass
    assert saved.dof_damping[0] == expected_damping
    verified = _verify_serialized_model(saved, full_precision)
    assert verified["comparison_tolerance"] == {"rtol": 1e-12, "atol": 1e-12}
    assert verified["scientific_thresholds_changed"] is False
    assert verified["fields"]["body_mass"]["maximum_absolute_serialization_error"] == abs(mass-expected_mass)
    json.dumps(verified, allow_nan=False)


def test_serialized_verification_rejects_stale_initial_model(tmp_path):
    study = load_study_config(ROOT / "configs/evidence_v1.json")
    initial = compile_model(ROOT / "model/pendulum.xml", study.nominal)
    fitted = compile_model(ROOT / "model/pendulum.xml", replace(study.nominal, mass_kg=1.1998996613593147))
    with pytest.raises(AssertionError):
        _verify_serialized_model(initial, fitted)


def test_official_saver_preserves_full_precision_for_non_six_digit_candidate(tmp_path, reduced_single_parameter_fit):
    """Serialization-only fixture; the supplied candidate is not a fitted result."""
    problem, _ = reduced_single_parameter_fit
    candidate = problem.params.copy()
    exact_mass = 1.1998996613593147
    candidate.update_from_vector(np.array([exact_mass]))
    residual = problem.evaluate(np.array([exact_mass]))
    execution = SimpleNamespace(
        optimized_params=candidate,
        optimize_result=OptimizeResult(x=np.array([exact_mass]), jac=np.ones((residual.size, 1))),
    )
    output = tmp_path / "serialization_fixture"
    info = save_official_fit(output, problem, execution, title="Serialization fixture; not an optimization result")
    saved_xml = mujoco.MjModel.from_xml_path(str(output / "pendulum.xml"))
    saved_binary = mujoco.MjModel.from_binary_path(str(output / "pendulum.mjb"))
    assert saved_xml.body_mass[1] == 1.1999
    assert saved_binary.body_mass[1] == exact_mass
    fields = info["verified_xml"]["pendulum"]["fields"]
    assert fields["body_mass"]["maximum_absolute_serialization_error"] == abs(exact_mass-1.1999)
    assert fields["body_mass"]["maximum_relative_serialization_error"] == abs(exact_mass-1.1999)/exact_mass
    assert info["parameter_precision"]["scientific_thresholds_changed"] is False
    assert "pendulum.mjb" in info["parameter_precision"]["full_precision_files"]


def test_optimization_summary_has_no_physical_truth_or_recovery_errors():
    problem, execution = diagnostic_fixture()
    diagnostic = summarize_optimization(problem, execution)
    assert diagnostic["physical_truth_evaluated"] is False
    assert diagnostic["schema_version"] == "mujoco-sysid-lab.optimization-summary.v1"
    for row in diagnostic["parameter_rows"]:
        assert not {"truth", "absolute_error", "relative_error", "error_reference"} & set(row)
    evaluated = summarize_fit(problem, execution, {"mass_kg": 1.2})
    assert "physical_truth_evaluated" not in evaluated
    for key in ("objective", "optimizer", "call_counts", "objective_trace", "runtime_seconds", "sensitivity", "confidence_intervals"):
        assert diagnostic[key] == evaluated[key]
    assert evaluated["parameter_rows"][0]["truth"] == 1.2
    json.dumps(diagnostic, allow_nan=False)


def test_invalid_fit_scope_rejects_before_creating_output(tmp_path):
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="fit_scope"):
        save_official_fit(output, None, None, fit_scope="real_robot")
    assert not output.exists()


@pytest.fixture(scope="module")
def reduced_joint_fit():
    """Two-second noisy joint test, separate from registered B evidence."""
    study = load_study_config(ROOT / "configs/evidence_v1.json")
    truth = load_ground_truth(study.truth_config_path, study)
    generator_model = compile_model(ROOT / "model/pendulum.xml", truth.model)
    times = np.arange(1001) * study.timestep_s
    sequences = []
    for identity, name in enumerate(("train_multisine", "train_prbs")):
        controls = (0.6 * np.sin(2*np.pi*(0.7+identity*0.5)*times))[:, None]
        controls[times < study.zero_control_prefix_s] = 0
        clean = simulate(generator_model, times, controls)
        noisy = []
        for sensor_id, (sensor, sigma) in enumerate(((clean.qpos, 0.002), (clean.qvel, 0.020))):
            rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([101, 99, identity, sensor_id])))
            noisy.append(sensor + sigma * rng.standard_normal(sensor.shape))
        sequences.append(TrainingSequence(name, identity, times, controls, *noisy, np.array([0., 0., 0.])))
    training = TrainingDataset(101, tuple(sequences))
    # Only nominal physics goes to the fit builder; truth is generator-only.
    problem = build_problem(training, ROOT / "model/pendulum.xml", study.nominal, study,
                            ("mass_kg", "damping_nm_s_per_rad"),
                            {"mass_kg": 0.8, "damping_nm_s_per_rad": 0.04})
    execution = problem.run()
    return problem, execution


def test_reduced_joint_optimization_summary_has_two_dimensional_local_diagnostics(reduced_joint_fit):
    problem, execution = reduced_joint_fit
    summary = summarize_optimization(problem, execution)
    assert len(summary["parameter_rows"]) == 2
    assert {row["name"] for row in summary["parameter_rows"]} == {"mass_kg", "damping_nm_s_per_rad"}
    assert summary["sensitivity"]["rank"] == 2
    assert len(summary["sensitivity"]["singular_values"]) == 2
    assert np.isfinite(summary["sensitivity"]["condition_number"])
    assert summary["confidence_intervals"]["status"] == "available"
    assert np.asarray(summary["confidence_intervals"]["covariance"]).shape == (2, 2)
    assert np.asarray(summary["confidence_intervals"]["intervals"]).shape == (2, 2)
    evaluated = summarize_fit(problem, execution, {"mass_kg": 1.2, "damping_nm_s_per_rad": 0.15})
    assert all(row["relative_error"] <= 0.1 for row in evaluated["parameter_rows"])
    json.dumps(summary, allow_nan=False)


def test_reduced_joint_official_html_and_binary_use_both_fitted_parameters(tmp_path, reduced_joint_fit):
    problem, execution = reduced_joint_fit
    before = problem.spec.to_xml()
    saved_counts = deepcopy(execution.call_counts)
    output = tmp_path / "joint_official"
    info = save_official_fit(output, problem, execution, title="Reduced synthetic joint test",
                             fit_scope="joint_physical")
    assert info["original_spec_unchanged"] and problem.spec.to_xml() == before
    assert execution.call_counts == saved_counts
    assert info["reporting_call_counts"]["residual_calls"] == 3
    html = (output / "report.html").read_text()
    assert "mass and damping are jointly estimated" in html
    assert "no physical truth is supplied to optimization or start selection" in html
    assert "oracle-assisted single-parameter study" not in html
    binary = mujoco.MjModel.from_binary_path(str(output / "pendulum.mjb"))
    fitted_mass = float(execution.optimized_params["mass_kg"].value[0])
    fitted_damping = float(execution.optimized_params["damping_nm_s_per_rad"].value[0])
    assert binary.body_mass[1] == fitted_mass
    assert binary.dof_damping[0] == fitted_damping
    original = problem.spec.compile()
    np.testing.assert_allclose(binary.body_inertia[1] / fitted_mass,
                               original.body_inertia[1] / original.body_mass[1], rtol=1e-14)
    saved = mujoco.MjModel.from_xml_path(str(output / "pendulum.xml"))
    assert saved.body_mass[1] == float(format(fitted_mass, ".6g"))
    assert saved.dof_damping[0] == float(format(fitted_damping, ".6g"))
    json.dumps(info, allow_nan=False)
