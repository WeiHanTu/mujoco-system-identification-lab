"""Reduced-duration implementation sanity checks, never retained evidence fits."""

from dataclasses import replace
from pathlib import Path

import mujoco
from mujoco import sysid
import numpy as np
import pytest

from mujoco_sysid_lab.dataset import TrainingDataset, TrainingSequence
from mujoco_sysid_lab.excitation import generate_control
from mujoco_sysid_lab.experiment_config import load_study_config
from mujoco_sysid_lab.identification import build_problem
from mujoco_sysid_lab.simulation import compile_model, simulate


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "model" / "pendulum.xml"


@pytest.fixture(scope="module")
def study():
    return load_study_config(ROOT / "configs" / "evidence_v1.json")


@pytest.fixture(scope="module")
def true_physics(study):
    return replace(study.nominal, mass_kg=1.20, damping_nm_s_per_rad=0.15)


@pytest.fixture(scope="module")
def training(study, true_physics):
    # First 1.2 s of the authored controls, independently reset per trajectory.
    # This labeled test fixture is not one of the 82 scientific evidence fits.
    model = compile_model(MODEL, true_physics)
    sequences = []
    for trajectory_id, name in enumerate(("train_multisine", "train_prbs")):
        times, controls, _ = generate_control(study, name)
        count = 601
        rollout = simulate(model, times[:count], controls[:count])
        sequences.append(TrainingSequence(name, trajectory_id, rollout.timestamps,
                                           rollout.controls, rollout.qpos, rollout.qvel,
                                           np.array([0.0, 0.0, 0.0])))
    return TrainingDataset(101, tuple(sequences))


def _fixed_nuisance(study, name):
    return replace(study.nominal, damping_nm_s_per_rad=0.15) if name == "mass_kg" else replace(
        study.nominal, mass_kg=1.20,
    )


def test_build_problem_rejects_slide_joint_before_constructing_estimator(tmp_path, study, training):
    altered = tmp_path/"unsupported_slide.xml"
    altered.write_text(MODEL.read_text().replace('type="hinge"', 'type="slide"'))
    with pytest.raises(ValueError, match="named hinge joint"):
        build_problem(training, altered, study.nominal, study,
                      ("mass_kg", "damping_nm_s_per_rad"),
                      {"mass_kg": .8, "damping_nm_s_per_rad": .04})


@pytest.mark.parametrize("name,truth", [("mass_kg", 1.20), ("damping_nm_s_per_rad", 0.15)])
def test_official_noiseless_single_parameter_recovery_with_fixed_nuisance(study, training, name, truth):
    physical = _fixed_nuisance(study, name)
    initial = getattr(study.nominal, name)
    problem = build_problem(training, MODEL, physical, study, (name,), {name: initial})
    pristine_xml = problem.spec.to_xml()
    result = problem.run()
    estimate = float(result.optimized_params[name].value[0])
    assert abs(estimate - truth) / truth <= study.criteria["noiseless_single_parameter_relative_error_max"]
    assert result.optimize_result.success
    assert problem.optimization_invoked is True
    assert result.optimize_result.status > 0
    assert result.optimize_result.nfev > 0 and result.optimize_result.njev > 0
    assert np.isfinite(result.runtime_seconds) and result.runtime_seconds > 0
    assert result.callback_events == sum(row["event"] == "callback" for row in result.objective_trace)
    assert result.objective_trace[0]["event"] == "initial"
    assert result.objective_trace[-1]["event"] == "final"
    assert not any("nit" in row for row in result.objective_trace)
    assert 0.5 * result.final_residual @ result.final_residual == pytest.approx(result.optimize_result.cost)
    np.testing.assert_array_equal(result.final_residual, result.optimize_result.fun)
    assert not result.initial_residual.flags.writeable and not result.final_residual.flags.writeable
    assert result.optimize_result.cost < 0.5 * result.initial_residual @ result.initial_residual
    assert problem.spec.to_xml() == pristine_xml
    np.testing.assert_array_equal(problem.params.as_vector(), [initial])
    assert len(result.final_predictions) == len(result.final_records) == 2
    assert all(len(group) == 1 for group in result.final_predictions)
    for phase in ("initial", "final"):
        assert result.call_counts[phase]["residual_calls"] == 1
        assert result.call_counts[phase]["logical_rollouts"] == 2
        assert result.call_counts[phase]["logical_perturbed_rollouts"] == 0
    optimization = result.call_counts["optimization"]
    assert optimization["residual_calls"] > result.optimize_result.nfev
    assert optimization["logical_perturbed_rollouts"] == 2 * result.optimize_result.njev
    assert optimization["logical_rollouts"] == 2 * optimization["candidate_columns"]
    before = dict(result.call_counts["total"])
    problem.evaluate(result.optimized_params.as_vector())
    assert result.call_counts["total"] == before
    assert problem.call_counts["external"]["residual_calls"] == 1


def test_known_noise_residual_sign_weighting_interleaving_and_safe_window(study, training, true_physics):
    sequences = []
    expected = []
    for index, sequence in enumerate(training.sequences):
        counts = np.arange(sequence.timestamps.size)
        q_offset = np.sin(counts * 0.19 + index)
        v_offset = np.cos(counts * 0.11 + index)
        sequences.append(replace(sequence,
            measured_qpos=sequence.measured_qpos + study.sensor_sigmas["qpos"] * q_offset[:, None],
            measured_qvel=sequence.measured_qvel + study.sensor_sigmas["qvel"] * v_offset[:, None]))
        lower, upper = study.prehistory_s, sequence.timestamps[-1] - study.prehistory_s
        mask = (sequence.timestamps >= lower - 1e-12) & (sequence.timestamps <= upper + 1e-12)
        expected.append(np.column_stack((q_offset[mask], v_offset[mask])).ravel())
    noisy = replace(training, sequences=tuple(sequences))
    problem = build_problem(noisy, MODEL, true_physics, study, ("mass_kg",), {"mass_kg": 1.20})
    residual = problem.evaluate(np.array([1.20]))
    np.testing.assert_allclose(residual, np.concatenate(expected), rtol=0, atol=1e-9)
    assert residual.size == 2 * 541 * 2
    assert 0.5 * residual @ residual == pytest.approx(0.5 * np.concatenate(expected) @ np.concatenate(expected))
    for measured in problem.sequences[0].sensordata:
        assert measured.times[0] == pytest.approx(.060)
        assert measured.times[-1] == pytest.approx(1.140)
        assert measured.signal_mapping.keys() == {"qpos", "qvel"}
    assert problem.sequences[0].control[0].signal_mapping.keys() == {"torque_ctrl"}


def test_mass_modifier_scales_inertia_and_preserves_fixed_damping_geometry_and_spec(study, training):
    physical = replace(study.nominal, damping_nm_s_per_rad=.15)
    problem = build_problem(training, MODEL, physical, study, ("mass_kg",), {"mass_kg": .8})
    pristine = problem.spec.to_xml()
    param = problem.params["mass_kg"]
    assert param.scale_rot_inertia is True
    candidate = problem.params.copy()
    candidate.update_from_vector(np.array([1.6]))
    model = sysid.apply_param_modifiers(candidate, problem.spec.copy())
    baseline = problem.spec.compile()
    np.testing.assert_array_equal(model.body("link").mass, [1.6])
    np.testing.assert_allclose(model.body("link").inertia, 2 * baseline.body("link").inertia)
    np.testing.assert_array_equal(model.body("link").ipos, baseline.body("link").ipos)
    np.testing.assert_array_equal(model.geom_size, baseline.geom_size)
    np.testing.assert_array_equal(model.dof_damping, [.15])
    np.testing.assert_array_equal(model.dof_dampingpoly, 0)
    assert problem.spec.to_xml() == pristine
    assert problem.params.keys() == ["mass_kg"]
    np.testing.assert_array_equal(problem.params.get_bounds(), [[.4], [2.]])


def test_damping_modifier_preserves_fixed_mass_and_all_inertia(study, training):
    physical = replace(study.nominal, mass_kg=1.2)
    problem = build_problem(training, MODEL, physical, study, ("damping_nm_s_per_rad",),
                            {"damping_nm_s_per_rad": .04})
    candidate = problem.params.copy()
    candidate.update_from_vector(np.array([.3]))
    model = sysid.apply_param_modifiers(candidate, problem.spec.copy())
    np.testing.assert_array_equal(model.body("link").mass, [1.2])
    np.testing.assert_array_equal(model.body("link").inertia, physical.inertia_at_com_kg_m2)
    np.testing.assert_array_equal(model.dof_damping, [.3])
    np.testing.assert_array_equal(model.dof_dampingpoly, 0)
    assert problem.params.keys() == ["damping_nm_s_per_rad"]
    np.testing.assert_array_equal(problem.params.get_bounds(), [[.001], [.5]])


def test_batched_official_finite_difference_shape_and_candidate_order(study, training):
    problem = build_problem(training, MODEL, _fixed_nuisance(study, "mass_kg"), study,
                            ("mass_kg",), {"mass_kg": .8})
    columns = np.array([[.7, 1.3]])
    batched, _, _ = problem.residual_fn(columns, problem.params.copy())
    for sequence_index, matrix in enumerate(batched):
        assert matrix.shape == (541 * 2, 2)
        for column in range(2):
            scalar, _, _ = problem.residual_fn(columns[:, column], problem.params.copy())
            np.testing.assert_array_equal(matrix[:, column], scalar[sequence_index])
    assert problem.call_counts["external"]["batched_residual_calls"] == 1
    assert problem.call_counts["external"]["logical_perturbed_rollouts"] == 4


def test_frozen_optimizer_options_and_bound_width_scaling(study, training):
    # Plumbing only: constructing a joint problem is not a joint evidence fit.
    problem = build_problem(training, MODEL, study.nominal, study,
                            ("damping_nm_s_per_rad", "mass_kg"),
                            {"mass_kg": .8, "damping_nm_s_per_rad": .04})
    assert problem.optimizer_options["max_iters"] == 200
    assert "max_nfev" not in problem.optimizer_options
    assert problem.optimizer_options["method"] == "trf"
    assert problem.optimizer_options["loss"] == "linear"
    assert problem.optimizer_options["diff_step"] == 1e-5
    np.testing.assert_array_equal(problem.optimizer_options["x_scale"], [.499, 1.6])
    assert problem.params.keys() == ["damping_nm_s_per_rad", "mass_kg"]


@pytest.mark.parametrize("free,start,error", [
    ((), {}, "unique mass"),
    (("mass_kg", "mass_kg"), {"mass_kg": .8}, "unique mass"),
    (("motor_gain",), {"motor_gain": 1.0}, "unique mass"),
    (("mass_kg",), {"damping_nm_s_per_rad": .04}, "exactly the free"),
    (("mass_kg",), {"mass_kg": .2}, "inside registered bounds"),
    (("mass_kg",), {"mass_kg": float("nan")}, "finite"),
    (("mass_kg",), {"mass_kg": True}, "finite"),
])
def test_invalid_free_names_and_starts_fail_before_optimizer(study, training, free, start, error):
    with pytest.raises(ValueError, match=error):
        build_problem(training, MODEL, study.nominal, study, free, start)


@pytest.mark.parametrize("changed,error", [
    (lambda sequence: replace(sequence, trajectory_name="heldout_chirp", trajectory_id=2), "two registered training"),
    (lambda sequence: replace(sequence, initial_state=np.array([0.0, .1, 0.0])), "equilibrium initial"),
    (lambda sequence: replace(sequence, measured_qpos=np.full((601, 1), np.nan)), "finite"),
    (lambda sequence: replace(sequence, controls=np.ones((601, 1)) * 1.6), "torque limit"),
    (lambda sequence: replace(sequence, timestamps=sequence.timestamps + .002), "complete fixed grid"),
    (lambda sequence: replace(sequence, timestamps=sequence.timestamps.tolist()), "complete fixed grid"),
    (lambda sequence: replace(sequence, measured_qpos=np.full((601, 1), "bad")), "finite"),
])
def test_unsafe_training_inputs_fail_before_official_rollout(study, training, changed, error):
    invalid = replace(training, sequences=(changed(training.sequences[0]), training.sequences[1]))
    with pytest.raises(ValueError, match=error):
        build_problem(invalid, MODEL, study.nominal, study, ("mass_kg",), {"mass_kg": .8})


def test_wrong_interface_and_fixed_mechanics_are_rejected(study, training):
    with pytest.raises(ValueError, match="sanitized TrainingDataset"):
        build_problem({"truth": 1.2}, MODEL, study.nominal, study, ("mass_kg",), {"mass_kg": .8})
    with pytest.raises(ValueError, match="changes fixed link_length_m"):
        build_problem(training, MODEL, replace(study.nominal, link_length_m=.7), study,
                      ("mass_kg",), {"mass_kg": .8})
    with pytest.raises(ValueError, match="schema version"):
        build_problem(replace(training, dataset_schema_version=2), MODEL, study.nominal, study,
                      ("mass_kg",), {"mass_kg": .8})


@pytest.mark.parametrize("candidate,error", [
    (np.array([np.nan]), "non-finite"),
    (np.array([.3]), "outside registered bounds"),
    (np.array([2.01]), "outside registered bounds"),
    (np.array([.8, .04]), "invalid dimensions"),
    (np.array([[.8]]), "flat single-candidate"),
])
def test_invalid_candidate_rejected_before_rollout(study, training, candidate, error):
    problem = build_problem(training, MODEL, study.nominal, study, ("mass_kg",), {"mass_kg": .8})
    with pytest.raises(ValueError, match=error):
        problem.evaluate(candidate)
    assert problem.call_counts["total"]["residual_calls"] == 0


def test_optimizer_failure_retains_actual_invocation_trace_and_counts(study, training, monkeypatch):
    problem = build_problem(training, MODEL, _fixed_nuisance(study, "mass_kg"), study,
                            ("mass_kg",), {"mass_kg": .8})

    def failing_optimize(*, initial_params, residual_fn, callback, **options):
        residual_fn(initial_params.as_vector(), initial_params)
        raise RuntimeError("injected official optimizer failure")

    monkeypatch.setattr(sysid, "optimize", failing_optimize)
    with pytest.raises(RuntimeError, match="official optimizer failure"):
        problem.run()
    assert problem.optimization_invoked is True
    assert [row["event"] for row in problem.last_trace] == ["initial"]
    assert problem.call_counts["initial"]["residual_calls"] == 1
    assert problem.call_counts["optimization"]["residual_calls"] == 1
    assert problem.call_counts["final"]["residual_calls"] == 0
    assert problem.residual_fn.phase == "external"
    np.testing.assert_array_equal(problem.params.as_vector(), [.8])


def test_initial_residual_failure_does_not_claim_optimizer_invocation(study, training, monkeypatch):
    problem = build_problem(training, MODEL, study.nominal, study, ("mass_kg",), {"mass_kg": .8})
    problem.optimization_invoked = True  # Must be reset even after an earlier run.

    def failing_initial(*args, **kwargs):
        raise RuntimeError("injected initial rollout failure")

    monkeypatch.setattr(problem.residual_fn, "official", failing_initial)
    monkeypatch.setattr(sysid, "optimize", lambda **kwargs: pytest.fail("optimizer must not be called"))
    with pytest.raises(RuntimeError, match="initial rollout failure"):
        problem.run()
    assert problem.optimization_invoked is False
    assert problem.last_trace == []
    assert problem.call_counts["initial"]["residual_calls"] == 1
    assert problem.call_counts["optimization"]["residual_calls"] == 0
    assert problem.residual_fn.phase == "external"


@pytest.fixture(scope="module")
def delayed_training(study, training):
    """Known 24 ms delay with real zero equilibrium history; no edge padding."""
    delay = .024
    dt = study.timestep_s
    prehistory = np.arange(-round(study.prehistory_s/dt), 0) * dt
    sequences = []
    for sequence in training.sequences:
        support = np.concatenate((prehistory, sequence.timestamps))
        query = sequence.timestamps - delay
        assert query.min() >= support[0] and query.max() <= support[-1]
        delayed = []
        for values in (sequence.measured_qpos, sequence.measured_qvel):
            clean = np.concatenate((np.zeros(prehistory.size), values[:, 0]))
            delayed.append(np.interp(query, support, clean)[:, None])
        sequences.append(replace(sequence, measured_qpos=delayed[0], measured_qvel=delayed[1]))
    return replace(training, sequences=tuple(sequences))


def test_shared_positive_delay_preserves_fixed_window_and_residual_shape(study, delayed_training, true_physics):
    problem = build_problem(delayed_training, MODEL, true_physics, study,
                            ("delay_s",), {"delay_s": .002})
    counts = []
    residuals = {}
    expected_times = [measured.times.copy() for measured in problem.sequences[0].sensordata]
    for delay in (0., .024, .060):
        residuals[delay] = problem.evaluate(np.array([delay]))
        counts.append(residuals[delay].size)
        _, predictions, records = problem.residual_fn(np.array([delay]), problem.params.copy(), return_pred_all=True)
        for index in range(2):
            np.testing.assert_array_equal(predictions[index][0].times, expected_times[index])
            np.testing.assert_array_equal(records[index][0].times, expected_times[index])
    assert counts == [2164, 2164, 2164]
    assert np.linalg.norm(residuals[.024]) < 1e-8
    assert np.linalg.norm(residuals[0.]) > 1
    assert np.linalg.norm(residuals[.060]) > 1
    assert problem.params["delay_s"].modifier is None
    np.testing.assert_array_equal(problem.optimizer_options["x_scale"], [.060])


def test_delay_excluded_model_is_exactly_zero_delay_baseline(study, delayed_training, true_physics):
    excluded = build_problem(delayed_training, MODEL, true_physics, study,
                             ("mass_kg",), {"mass_kg": 1.2})
    included = build_problem(delayed_training, MODEL, true_physics, study,
                             ("mass_kg", "delay_s"), {"mass_kg": 1.2, "delay_s": .002})
    np.testing.assert_array_equal(excluded.evaluate(np.array([1.2])),
                                  included.evaluate(np.array([1.2, 0.])))
    assert "delay_s" not in excluded.params.keys()


def test_delay_only_then_joint_noiseless_recovery_smoke(study, delayed_training, true_physics):
    # Separate reduced implementation fixture; fixed truth physics is not the
    # registered C delay-only fit, which must use that seed's immutable B fit.
    delay_only = build_problem(delayed_training, MODEL, true_physics, study,
                               ("delay_s",), {"delay_s": .002})
    delay_result = delay_only.run()
    assert abs(float(delay_result.optimized_params["delay_s"].value[0]) - .024) <= study.timestep_s
    assert delay_result.optimize_result.success
    joint = build_problem(delayed_training, MODEL, study.nominal, study,
                          ("mass_kg", "damping_nm_s_per_rad", "delay_s"),
                          {"mass_kg": .8, "damping_nm_s_per_rad": .04, "delay_s": .002})
    result = joint.run()
    assert result.optimize_result.success
    for name, truth in (("mass_kg", 1.2), ("damping_nm_s_per_rad", .15)):
        assert abs(float(result.optimized_params[name].value[0]) - truth)/truth <= .001
    assert abs(float(result.optimized_params["delay_s"].value[0]) - .024) <= study.timestep_s
    assert result.optimize_result.jac.shape == (2164, 3)
    assert result.final_residual.shape == result.initial_residual.shape


@pytest.mark.parametrize("delay", [-.001, .060001, float("nan"), True])
def test_delay_start_outside_bound_or_nonfinite_is_rejected(study, training, delay):
    with pytest.raises(ValueError, match="start delay_s must be finite and inside"):
        build_problem(training, MODEL, study.nominal, study, ("delay_s",), {"delay_s": delay})


def test_unsafe_registered_delay_support_fails_without_candidate_dependent_cropping(study, training):
    from mujoco_sysid_lab.config import ParameterBounds
    unsafe = replace(study, delay_bounds_s=ParameterBounds(0., .08))
    problem = build_problem(training, MODEL, study.nominal, unsafe, ("delay_s",), {"delay_s": .002})
    with pytest.raises(ValueError, match="unsafe for delay bounds"):
        problem.evaluate(np.array([.002]))


def test_three_parameter_order_batched_residual_and_delay_do_not_modify_dynamics(study, delayed_training):
    problem = build_problem(delayed_training, MODEL, study.nominal, study,
                            ("delay_s", "damping_nm_s_per_rad", "mass_kg"),
                            {"mass_kg": .8, "damping_nm_s_per_rad": .04, "delay_s": .002})
    assert problem.params.keys() == ["delay_s", "damping_nm_s_per_rad", "mass_kg"]
    np.testing.assert_array_equal(problem.optimizer_options["x_scale"], [.06, .499, 1.6])
    columns = np.array([[.0, .024], [.15, .15], [1.2, 1.2]])
    batched, _, _ = problem.residual_fn(columns, problem.params.copy())
    for index, matrix in enumerate(batched):
        for column in range(2):
            scalar, _, _ = problem.residual_fn(columns[:, column], problem.params.copy())
            np.testing.assert_array_equal(matrix[:, column], scalar[index])
    candidates = []
    for column in range(2):
        params = problem.params.copy()
        params.update_from_vector(columns[:, column])
        candidates.append(sysid.apply_param_modifiers(params, problem.spec.copy()))
    for field in ("body_mass", "body_inertia", "body_ipos", "dof_damping"):
        np.testing.assert_array_equal(getattr(candidates[0], field), getattr(candidates[1], field))
    assert candidates[0].body_mass[1] == 1.2
    assert candidates[0].dof_damping[0] == .15
    assert candidates[0].dof_frictionloss[0] == 0


def test_weak_training_requires_explicit_whitelist_and_one_rollout(study, true_physics):
    model = compile_model(MODEL, true_physics)
    times, controls, _ = generate_control(study, "weak_step")
    clean = simulate(model, times[:601], controls[:601])
    sequence = TrainingSequence("weak_step", 4, clean.timestamps, clean.controls,
                                clean.qpos, clean.qvel, np.array([0., 0., 0.]))
    weak = TrainingDataset(101, (sequence,))
    with pytest.raises(ValueError, match="two registered training"):
        build_problem(weak, MODEL, study.nominal, study, ("mass_kg",), {"mass_kg": .8})
    problem = build_problem(weak, MODEL, study.nominal, study,
                            ("mass_kg", "damping_nm_s_per_rad"),
                            {"mass_kg": .8, "damping_nm_s_per_rad": .04}, training_ids=("weak_step",))
    assert problem.training_ids == ("weak_step",)
    assert problem.sequences[0].sequence_name == ["weak_step"]
    assert problem.sequences[0].control[0].data.shape == (601, 1)
    assert problem.sequences[0].sensordata[0].data.shape == (541, 2)
    assert problem.evaluate(np.array([1.2, .15])).shape == (1082,)
    assert problem.call_counts["external"]["logical_rollouts"] == 1


@pytest.mark.parametrize("selected", [("heldout_chirp",), ("train_multisine",), (), ["weak_step"]])
def test_explicit_training_selection_cannot_authorize_heldout_or_partial_rich_data(study, training, selected):
    with pytest.raises(ValueError, match="training_ids"):
        build_problem(training, MODEL, study.nominal, study,
                      ("mass_kg",), {"mass_kg": .8}, training_ids=selected)
