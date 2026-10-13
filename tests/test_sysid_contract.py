"""Executable release contracts, not scientific parameter-recovery claims."""

import importlib
import json
from pathlib import Path

import mujoco
from mujoco import sysid
import numpy as np
import pytest

from mujoco_sysid_lab.api_check import check_api, make_aligned_observation_hook
from mujoco_sysid_lab.config import load_config
from mujoco_sysid_lab.simulation import load_spec, simulate


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def model():
    config = load_config(ROOT / "configs/milestone1.json")
    return load_spec(ROOT / "model/pendulum.xml", config.ground_truth).compile()


@pytest.fixture(scope="module")
def api_record():
    config = load_config(ROOT / "configs/milestone1.json")
    return check_api(ROOT / "model/pendulum.xml", config.ground_truth)


def test_released_api_and_strict_source_record(api_record):
    assert api_record["mujoco_version"] == "3.15.0"
    assert api_record["required_exports_present"]
    json.dumps(api_record, allow_nan=False)
    for record in api_record["source_records"].values():
        assert record["source_path"].startswith("mujoco/")
        assert len(record["source_sha256"]) == 64
        assert "0x" not in record["signature"]
    assert not api_record["optimizer_inspection"]["executed_estimation"]
    assert not api_record["report_inspection"]["executed_result_save_or_html_generation"]


def test_physical_modifiers_scale_inertia_and_skip_frozen(api_record):
    checks = api_record["physical_modifier_checks"]
    assert checks["mass_and_inertia_scaled_together"]
    assert checks["inertia_ratio_max_abs_error"] < 1e-14
    assert checks["com_unchanged"]
    assert checks["frozen_physical_modifier_skipped"]
    assert checks["mjsjoint_damping_shape"] == [3]


def test_official_pipeline_timing_and_noise_weight_equivalence(api_record):
    checks = api_record["pipeline_checks"]
    assert checks["official_pipeline_executed_without_optimization"]
    assert checks["upstream_sensors_equal_pre_step"]
    assert checks["upstream_sensor_timestamp_offset_s"] == 0.002
    assert checks["upstream_post_step_sensor_max_abs_error"] > 1e-3
    assert checks["aligned_identity_residual_max_abs"] < 1e-10
    assert checks["noise_weighted_residual_max_abs_error"] < 1e-10
    assert checks["residual_sign"] == "measured_minus_predicted"


def test_delay_sign_and_candidate_independent_safe_mask(api_record):
    checks = api_record["delay_checks"]
    assert checks["upstream_unwindowed_unsafe_samples_at_max_delay"] == 30
    assert checks["positive_delay_samples_t_minus_delay"]
    assert checks["safe_window_candidate_independent"]
    assert checks["safe_query_support_verified"]
    assert checks["monotone_fixture_max_abs_error"] < 1e-14


def _monotone_fixture(model):
    times = np.arange(501, dtype=np.float64) * 0.002
    state = np.column_stack((times[1:], times[1:], 2 * times[1:] + 1))
    initial = np.array([0.0, 0.0, 1.0])
    sensors = sysid.TimeSeries.from_names(
        times, np.column_stack((times, 2 * times + 1)), model, ["qpos", "qvel"],
    )
    safe = (times >= 0.060) & (times <= 0.940)
    query = times[safe] - 0.024
    measured = sysid.TimeSeries.from_names(
        times[safe], np.column_stack((query, 2 * query + 1)), model, ["qpos", "qvel"],
    )
    params = sysid.ParameterDict()
    params.add(sysid.Parameter("delay", 0.024, 0.0, 0.060))
    hook = make_aligned_observation_hook(initial, {"qpos": 1.0, "qvel": 1.0}, params["delay"])
    return sensors, measured, params, hook, state


@pytest.mark.parametrize("candidate", [0.0, 0.024, 0.060])
def test_aligned_hook_delay_extremes_retain_samples(model, candidate):
    sensors, measured, params, hook, state = _monotone_fixture(model)
    params.update_from_vector(np.array([candidate]))
    residual, predicted, returned = hook(params, sensors, measured, model, False, state=state)
    np.testing.assert_array_equal(returned.times, measured.times)
    np.testing.assert_array_equal(predicted.times, measured.times)
    np.testing.assert_allclose(residual, np.broadcast_to(
        [candidate - 0.024, 2 * (candidate - 0.024)], residual.shape), rtol=0, atol=1e-14)


def test_aligned_hook_rejects_unsafe_window_and_out_of_bounds_delay(model):
    sensors, measured, params, hook, state = _monotone_fixture(model)
    with pytest.raises(ValueError, match="unsafe for delay bounds"):
        hook(params, sensors, sensors, model, False, state=state)
    params.update_from_vector(np.array([0.061]))
    with pytest.raises(ValueError, match="outside registered bounds"):
        hook(params, sensors, measured, model, False, state=state)


@pytest.mark.parametrize("initial, scales", [
    (np.array([0.0, 0.0]), {"qpos": 0.002, "qvel": 0.020}),
    (np.array([0.002, 0.0, 0.0]), {"qpos": 0.002, "qvel": 0.020}),
    (np.array([0.0, np.nan, 0.0]), {"qpos": 0.002, "qvel": 0.020}),
    (np.zeros(3), {"qpos": 0.0, "qvel": 0.020}),
    (np.zeros(3), {"qpos": 0.002, "qvel": np.inf}),
    (np.zeros(3), {"qpos": 0.002, "velocity": 0.020}),
])
def test_hook_rejects_invalid_initial_state_or_noise_scales(initial, scales):
    with pytest.raises(ValueError):
        make_aligned_observation_hook(initial, scales)


@pytest.mark.parametrize("bounds", [(-0.01, 0.060), (0.060, 0.0), (0.01, 0.01), (0.0, np.inf)])
def test_hook_rejects_invalid_delay_ranges(bounds):
    delay = sysid.Parameter("delay", 0.024, *bounds)
    with pytest.raises(ValueError, match="delay bounds"):
        make_aligned_observation_hook(np.zeros(3), {"qpos": 0.002, "qvel": 0.020}, delay)


def test_hook_rejects_missing_or_invalid_fullphysics_state(model):
    sensors, measured, params, hook, state = _monotone_fixture(model)
    for bad_state in (None, state[:, :2], np.full_like(state, np.nan), state[::-1]):
        with pytest.raises(ValueError, match="state|timestamps"):
            hook(params, sensors, measured, model, False, state=bad_state)
    shifted = state.copy()
    shifted[:, 0] += 0.002
    with pytest.raises(ValueError, match="fixed simulation grid"):
        hook(params, sensors, measured, model, False, state=shifted)


def test_hook_rejects_other_sensor_semantics():
    config = load_config(ROOT / "configs/milestone1.json")
    spec = load_spec(ROOT / "model/pendulum.xml", config.ground_truth)
    spec.sensor("qpos").name = "position"
    changed = spec.compile()
    sensors = sysid.TimeSeries.from_names(np.array([0.0, 0.002]), np.zeros((2, 2)), changed)
    hook = make_aligned_observation_hook(np.zeros(3), {"qpos": 0.002, "qvel": 0.020})
    with pytest.raises(ValueError, match="named jointpos/qvel"):
        hook(sysid.ParameterDict(), sensors, sensors, changed, False, state=np.zeros((1, 3)))


def test_hook_rejects_slide_joint():
    config = load_config(ROOT / "configs/milestone1.json")
    spec = load_spec(ROOT / "model/pendulum.xml", config.ground_truth)
    spec.joint("hinge").type = mujoco.mjtJoint.mjJNT_SLIDE
    changed = spec.compile()
    sensors = sysid.TimeSeries.from_names(np.array([0.0, 0.002]), np.zeros((2, 2)), changed)
    hook = make_aligned_observation_hook(np.zeros(3), {"qpos": 0.002, "qvel": 0.020})
    with pytest.raises(ValueError, match="exactly one hinge"):
        hook(sysid.ParameterDict(), sensors, sensors, changed, False, state=np.zeros((1, 3)))


def test_eight_second_clock_roundoff_keeps_fixed_delay_boundaries(model):
    """Timing verification only: no saved study dataset or estimation."""
    times = np.arange(4001, dtype=np.float64) * model.opt.timestep
    controls = (0.45 * np.sin(np.arange(times.size) * 0.37))[:, None]
    direct = simulate(model, times, controls, initial_qpos=0.17, initial_qvel=0.23)
    clean = sysid.TimeSeries.from_names(times, direct.sensordata, model, ["qpos", "qvel"])
    control = sysid.TimeSeries.from_control_names(times, controls, model, ["torque"])
    initial = sysid.create_initial_state(model, np.array([0.17]), np.array([0.23]))
    official = sysid.sysid_rollout(model, mujoco.MjData(model), control, initial)[0]
    clock_error = float(np.max(np.abs(official.state.data[:, 0] - times[1:])))
    assert 0 < clock_error < 1e-10
    safe = (times >= 0.060) & (times <= 7.940)
    measured = sysid.apply_resample_and_delay(clean, times[safe], 0.060)
    params = sysid.ParameterDict()
    params.add(sysid.Parameter("delay", 0.060, 0.0, 0.060))
    hook = make_aligned_observation_hook(initial, {"qpos": 0.002, "qvel": 0.020}, params["delay"])
    residual, predicted, returned = hook(params, official.sensordata, measured, model, False, state=official.state.data)
    assert returned.times[0] == 0.060 and returned.times[-1] == 7.940
    np.testing.assert_array_equal(predicted.times, measured.times)
    np.testing.assert_allclose(residual, 0.0, rtol=0, atol=1e-10)


def test_official_batched_finite_difference_is_bounded(api_record):
    checks = api_record["finite_difference_checks"]
    assert checks["official_batched_fd_bounds_respected"]
    assert checks["analytic_jacobian_max_abs_error"] < 5e-8
    assert checks["reported_added_residual_evaluations"] == 2
    assert checks["python_residual_calls"] == 2


def test_official_scipy_wrapper_maps_budget_and_fd_without_fitting(monkeypatch):
    """Intercept backend dispatch: check options, never call an optimizer."""
    optimize_module = importlib.import_module("mujoco.sysid._src.optimize")
    captured = {}
    marker = object()

    def capture_least_squares(residual_fn, x0, **kwargs):
        captured.update(kwargs)
        np.testing.assert_allclose(kwargs["jac"](x0), [[2.0]], rtol=0, atol=1e-7)
        return marker

    monkeypatch.setattr(optimize_module.scipy_optimize, "least_squares", capture_least_squares)
    result = optimize_module._scipy_least_squares(
        np.array([0.4]), lambda x: 2 * x,
        (np.array([0.4]), np.array([2.0])), use_mujoco_jac=True,
        max_iters=7, verbose=False, diff_step=1e-6, method="trf", loss="linear", x_scale=np.array([1.6]),
    )
    assert result is marker
    assert captured["max_nfev"] == 7
    assert captured["verbose"] == 0
    assert captured["method"] == "trf"
    assert captured["loss"] == "linear"
    assert "max_iters" not in captured and "diff_step" not in captured


def test_stale_signal_transform_keyword_fails_without_rollout(model):
    residual_fn = sysid.build_residual_fn(models_sequences=[], signal_transform=sysid.SignalTransform())
    with pytest.raises(TypeError, match="signal_transform"):
        residual_fn(np.array([]), sysid.ParameterDict())
