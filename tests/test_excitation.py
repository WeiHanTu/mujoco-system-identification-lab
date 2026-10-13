from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from mujoco_sysid_lab.excitation import generate_control
from mujoco_sysid_lab.experiment_config import CORE_TRAJECTORY_IDS, load_study_config


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def study():
    return load_study_config(ROOT / "configs" / "evidence_v1.json")


@pytest.mark.parametrize("name", (*CORE_TRAJECTORY_IDS, "weak_step"))
def test_controls_deterministic_finite_and_on_grid(study, name):
    times, controls, info = generate_control(study, name)
    repeated = generate_control(study, name)
    np.testing.assert_array_equal(times, repeated[0])
    np.testing.assert_array_equal(controls, repeated[1])
    assert info == repeated[2]
    assert np.isfinite(times).all() and np.isfinite(controls).all()
    assert controls.shape == (times.size, 1)
    assert times[-1] == study.trajectories[name].duration_s
    np.testing.assert_array_equal(times, np.arange(times.size)*study.timestep_s)
    assert np.all(controls[:50] == 0)
    assert controls[-1, 0] == 0
    assert info["terminal_control_unused"] is True
    assert info["torque_peak_abs_nm"] <= 1.5
    assert info["sampled_torque_energy_nm2_s"] == pytest.approx(np.sum(controls[:-1, 0]**2)*study.timestep_s)
    assert info["torque_rms_nm"] == pytest.approx(np.sqrt(np.mean(controls[:-1, 0]**2)))
    assert info["torque_dc_nm"] == pytest.approx(np.mean(controls[:-1, 0]))
    assert json.dumps(info, allow_nan=False)
    assert not times.flags.writeable and not controls.flags.writeable


def test_multisine_rms_and_active_phase_reference(study):
    _, control, info = generate_control(study, "train_multisine")
    assert info["active_torque_rms_nm"] == pytest.approx(.55, abs=1e-15)
    assert info["sampled_torque_energy_nm2_s"] == pytest.approx(.55**2*7.9)
    raw_first = sum(amplitude*np.sin(phase) for amplitude, phase in zip([.2,.3,.25,.15], [0,.7,1.3,2.1]))
    assert control[50, 0] == pytest.approx(raw_first*info["rms_rescale_factor"])
    assert info["frequencies_hz"] == [.25,.65,1.15,1.8]


def test_prbs_exact_seed_block_and_rms_contract(study):
    _, control, info = generate_control(study, "train_prbs")
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([314159, 100, 1])))
    expected_levels = rng.choice([-1., -.6, 0., .6, 1.], size=32)
    assert info["seed_sequence"] == [314159, 100, 1]
    np.testing.assert_array_equal(info["raw_block_levels"], expected_levels)
    assert info["last_block_intervals"] == 75
    for index in range(32):
        start = 50 + index*125
        end = min(start+125, 4000)
        np.testing.assert_array_equal(control[start:end, 0], np.full(end-start, expected_levels[index]*info["rms_rescale_factor"]))
    assert info["active_torque_rms_nm"] == pytest.approx(.55, abs=1e-15)
    other_seed = replace(study, primary_seed=105)
    np.testing.assert_array_equal(generate_control(other_seed, "train_prbs")[1], control)


def test_chirp_onset_and_literal_sample(study):
    times, control, info = generate_control(study, "heldout_chirp")
    assert control[50, 0] == 0
    index = 75
    active_time = times[index]-.1
    expected = .8*np.sin(2*np.pi*(.2*active_time+.5*(1.8/7.9)*active_time**2))*.5*(1-np.cos(np.pi*active_time/.1))
    assert control[index, 0] == pytest.approx(expected)
    assert info["sweep_duration_s"] == 7.9
    assert np.max(np.abs(control)) <= .8


def test_pulses_use_half_open_interval_boundaries(study):
    _, control, info = generate_control(study, "heldout_pulse")
    for start, end, amplitude in ((250, 600, 1.25), (1250, 1650, -.90), (2500, 2725, .95)):
        assert control[start-1, 0] == 0
        np.testing.assert_array_equal(control[start:end, 0], np.full(end-start, amplitude))
        assert control[end, 0] == 0
    assert info["sampled_torque_energy_nm2_s"] == pytest.approx(1.25**2*.7+.9**2*.8+.95**2*.45)


def test_weak_step_records_single_reset_energy_difference(study):
    _, controls, weak = generate_control(study, "weak_step")
    assert controls.shape == (8001, 1)
    np.testing.assert_array_equal(controls[50:-1, 0], np.full(7950, .55))
    assert weak["active_torque_rms_nm"] == .55
    assert weak["sampled_torque_energy_nm2_s"] == pytest.approx(.55**2*15.9)
    rich_energy = sum(generate_control(study, name)[2]["sampled_torque_energy_nm2_s"] for name in CORE_TRAJECTORY_IDS[:2])
    assert weak["sampled_torque_energy_nm2_s"]-rich_energy == pytest.approx(.55**2*.1)


def test_over_limit_signal_is_rejected_without_clipping(study):
    low_limit = replace(study, nominal=replace(study.nominal, control_limit_nm=.5))
    with pytest.raises(ValueError, match="exceeds registered.*no clipping"):
        generate_control(low_limit, "train_multisine")
    with pytest.raises(ValueError, match="unregistered trajectory"):
        generate_control(study, "not_registered")


def test_control_to_torque_known_gain_conversion(study):
    _, normal, info = generate_control(study, "heldout_pulse")
    _, control, changed_info = generate_control(replace(study, nominal=replace(study.nominal, motor_gain_nm_per_unit=2)), "heldout_pulse")
    np.testing.assert_array_equal(control*2, normal)
    assert changed_info["sampled_torque_energy_nm2_s"] == info["sampled_torque_energy_nm2_s"]
