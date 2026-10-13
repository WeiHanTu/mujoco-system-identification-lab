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
    (("delay_s",), {"delay_s": .024}, "unique mass"),
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
