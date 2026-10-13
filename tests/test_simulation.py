from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import pytest

from mujoco_sysid_lab.config import load_config
from mujoco_sysid_lab.simulation import compile_model, simulate


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "model" / "pendulum.xml"


@pytest.fixture
def config():
    return load_config(PROJECT_ROOT / "configs" / "milestone1.json")


@pytest.fixture
def model(config):
    return compile_model(MODEL_PATH, config.ground_truth)


def test_compiled_model_is_one_dof_and_matches_fixed_mechanics(model, config):
    expected = config.ground_truth
    assert (model.nq, model.nv, model.nu, model.nsensor, model.nsensordata) == (1, 1, 1, 2, 2)
    assert model.opt.timestep == expected.timestep_s
    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    np.testing.assert_array_equal(model.opt.gravity, [0, 0, -9.81])
    link_id = model.body("link").id
    assert model.body_mass[link_id] == expected.mass_kg
    np.testing.assert_array_equal(model.body_inertia[link_id], expected.inertia_at_com_kg_m2)
    np.testing.assert_array_equal(model.body_ipos[link_id], [0, 0, -0.25])
    np.testing.assert_array_equal(model.dof_damping, [expected.damping_nm_s_per_rad])
    np.testing.assert_array_equal(model.dof_dampingpoly, np.zeros_like(model.dof_dampingpoly))
    np.testing.assert_array_equal(model.dof_armature, [0])
    np.testing.assert_array_equal(model.dof_frictionloss, [0])
    np.testing.assert_array_equal(model.actuator_gear[0], [1, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(model.actuator_ctrlrange[0], [-1.5, 1.5])
    assert not model.actuator_ctrllimited[0]
    np.testing.assert_array_equal(model.geom_contype, 0)
    np.testing.assert_array_equal(model.geom_conaffinity, 0)
    assert model.joint("hinge").type[0] == mujoco.mjtJoint.mjJNT_HINGE
    np.testing.assert_array_equal(model.joint("hinge").axis, [0, 1, 0])


def test_compiling_changed_mass_scales_all_inertia_axes(config):
    original = compile_model(MODEL_PATH, config.ground_truth)
    doubled = compile_model(MODEL_PATH, replace(config.ground_truth, mass_kg=2.4))
    np.testing.assert_array_equal(doubled.body_mass[1], original.body_mass[1] * 2)
    np.testing.assert_array_equal(doubled.body_inertia[1], original.body_inertia[1] * 2)
    np.testing.assert_array_equal(doubled.body_ipos, original.body_ipos)
    np.testing.assert_array_equal(doubled.geom_size, original.geom_size)


def test_equilibrium_is_exact_with_zero_torque(model):
    times = np.arange(501) * model.opt.timestep
    trajectory = simulate(model, times, np.zeros((times.size, 1)))
    np.testing.assert_array_equal(trajectory.qpos, np.zeros((times.size, 1)))
    np.testing.assert_array_equal(trajectory.qvel, np.zeros((times.size, 1)))
    np.testing.assert_array_equal(trajectory.sensordata, np.zeros((times.size, 2)))


def test_step_uses_interval_control_and_records_fresh_sensors(model):
    dt = model.opt.timestep
    times = np.arange(6) * dt
    controls = np.array([0, 1, 0, 0, 0, -1.5])
    trajectory = simulate(model, times, controls)
    assert trajectory.qpos.shape == trajectory.qvel.shape == trajectory.controls.shape == (6, 1)
    assert trajectory.sensordata.shape == (6, 2)
    np.testing.assert_array_equal(trajectory.qpos[:2], [[0], [0]])
    np.testing.assert_array_equal(trajectory.qvel[:2], [[0], [0]])
    assert trajectory.qvel[2, 0] > 0
    # At q=0, gravity torque is zero; implicit viscous damping gives this
    # first driven-step velocity. This checks timing and stamped mechanics.
    inertia = model.body_inertia[1, 1] + model.body_mass[1] * model.body_ipos[1, 2]**2
    expected_velocity = dt / (inertia + dt * model.dof_damping[0])
    assert trajectory.qvel[2, 0] == pytest.approx(expected_velocity, rel=1e-13)
    assert trajectory.qpos[2, 0] == pytest.approx(dt * expected_velocity, rel=1e-13)
    np.testing.assert_array_equal(trajectory.sensordata[:, :1], trajectory.qpos)
    np.testing.assert_array_equal(trajectory.sensordata[:, 1:], trajectory.qvel)
    changed_terminal = controls.copy()
    changed_terminal[-1] = 1.5
    same_states = simulate(model, times, changed_terminal)
    np.testing.assert_array_equal(same_states.qpos, trajectory.qpos)
    np.testing.assert_array_equal(same_states.qvel, trajectory.qvel)


def test_fixed_seed_control_and_simulation_are_exactly_repeatable(model):
    times = np.arange(501) * model.opt.timestep
    controls = np.random.Generator(np.random.PCG64(101)).uniform(-1, 1, (times.size, 1))
    first = simulate(model, times, controls)
    second = simulate(model, times, controls)
    for name in ("timestamps", "controls", "qpos", "qvel", "sensordata"):
        a, b = getattr(first, name), getattr(second, name)
        np.testing.assert_array_equal(a, b)
        assert np.isfinite(a).all()
        assert not a.flags.writeable
    assert not np.shares_memory(first.timestamps, times)
    assert not np.shares_memory(first.controls, controls)


def test_initial_state_is_explicit(model):
    trajectory = simulate(model, np.arange(3) * model.opt.timestep, np.zeros(3), initial_qpos=0.1, initial_qvel=-0.2)
    np.testing.assert_array_equal(trajectory.qpos[0], [0.1])
    np.testing.assert_array_equal(trajectory.qvel[0], [-0.2])
    np.testing.assert_array_equal(trajectory.sensordata[0], [0.1, -0.2])


@pytest.mark.parametrize("times,controls,error", [
    (np.array([0]), np.zeros(1), "at least two samples"),
    (np.array([[0, 0.002]]), np.zeros(2), "one-dimensional"),
    (np.array([0, 0.002]), np.zeros(1), "controls must have shape"),
    (np.array([0.002, 0.004]), np.zeros(2), "start.*0.0"),
    (np.array([0, 0.003]), np.zeros(2), "match the simulation timestep"),
    (np.array([0, 0]), np.zeros(2), "strictly increasing"),
    (np.array([0, np.nan]), np.zeros(2), "finite"),
    (np.array([0, 0.002]), np.array([0, np.inf]), "finite"),
    (np.array([0, 0.002]), np.array([1.50001, 0]), "torque limit"),
])
def test_invalid_trajectory_inputs_are_rejected(model, times, controls, error):
    with pytest.raises(ValueError, match=error):
        simulate(model, times, controls)


def test_nonfinite_initial_state_is_rejected(model):
    with pytest.raises(ValueError, match="initial qpos and qvel must be finite"):
        simulate(model, np.array([0, 0.002]), np.zeros(2), initial_qpos=float("nan"))


@pytest.mark.parametrize("change,error", [
    ("reordered_sensors", "sensors must be named"),
    ("activation", "na=0"),
    ("slide", "named hinge joint"),
])
@pytest.mark.parametrize("entry_point", ["compile", "simulate"])
def test_matching_dimensions_do_not_admit_unsupported_model_semantics(tmp_path, config, change, error, entry_point):
    xml = MODEL_PATH.read_text()
    if change == "reordered_sensors":
        xml = xml.replace(
            '<jointpos name="qpos" joint="hinge"/>\n    <jointvel name="qvel" joint="hinge"/>',
            '<jointvel name="qvel" joint="hinge"/>\n    <jointpos name="qpos" joint="hinge"/>',
        )
    elif change == "activation":
        xml = xml.replace(
            '<motor name="torque" joint="hinge" gear="1" ctrllimited="false"/>',
            '<general name="torque" joint="hinge" gear="1" dyntype="filter" dynprm="0.1" ctrllimited="false"/>',
        )
    else:
        xml = xml.replace('type="hinge"', 'type="slide"')
    assert xml != MODEL_PATH.read_text()
    if entry_point == "compile":
        altered = tmp_path/"unsupported.xml"
        altered.write_text(xml)
        with pytest.raises(ValueError, match=error):
            compile_model(altered, config.ground_truth)
    else:
        unsupported = mujoco.MjModel.from_xml_string(xml)
        assert (unsupported.nq, unsupported.nv, unsupported.nu, unsupported.nsensordata) == (1, 1, 1, 2)
        unsupported.actuator_ctrlrange[0] = [-1.5, 1.5]
        with pytest.raises(ValueError, match=error):
            simulate(unsupported, np.arange(2)*.002, np.zeros((2, 1)), initial_qpos=.1, initial_qvel=-.2)


@pytest.mark.parametrize("required_name", ["link", "hinge", "link_geometry", "torque"])
def test_missing_named_model_node_is_rejected_before_stamping(tmp_path, config, required_name):
    # Rename references too, so the MJCF is otherwise a valid one-joint model.
    xml = MODEL_PATH.read_text().replace(f'name="{required_name}"', f'name="other_{required_name}"')
    if required_name == "hinge":
        xml = xml.replace('joint="hinge"', 'joint="other_hinge"')
    altered = tmp_path/"renamed.xml"
    altered.write_text(xml)
    with pytest.raises(ValueError, match=f"requires named .*{required_name}"):
        compile_model(altered, config.ground_truth)
