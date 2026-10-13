"""Installed-release contracts and a narrow official observation hook.

No estimation runs in this module. MuJoCo 3.15.0's rollout sensors are evaluated
before integration but labelled at the post-step state time by SysID. For this
specific jointpos/jointvel model, the supported ``modify_residual`` hook reads
the corresponding fresh values from the explicit FULLPHYSICS state instead.
"""

from collections.abc import Callable, Mapping
from dataclasses import asdict
from hashlib import sha256
import importlib
import inspect
from pathlib import Path
import re

import mujoco
from mujoco import minimize, sysid
import numpy as np

from mujoco_sysid_lab.config import ModelConfig
from mujoco_sysid_lab.simulation import load_spec, simulate


REQUIRED_SYMBOLS = (
    "Parameter", "ParameterDict", "TimeSeries", "create_initial_state",
    "ModelSequences", "build_residual_fn", "optimize", "save_results",
    "default_report", "SignalTransform", "body_inertia_param",
    "apply_param_modifiers", "sysid_rollout", "apply_resample_and_delay",
    "apply_delayed_ts_window", "calculate_intervals",
)


def _source_record(obj: object) -> dict:
    """Keep package-relative paths and hashes without user home identifiers."""
    source_path = Path(inspect.getsourcefile(obj))
    package_parent = Path(mujoco.__file__).parent.parent
    return {
        "signature": re.sub(r" at 0x[0-9a-fA-F]+", "", str(inspect.signature(obj))),
        "source_path": source_path.relative_to(package_parent).as_posix(),
        "source_sha256": sha256(source_path.read_bytes()).hexdigest(),
    }


def make_aligned_observation_hook(
    initial_state: np.ndarray,
    sensor_sigmas: Mapping[str, float],
    delay_parameter: sysid.Parameter | None = None,
) -> Callable:
    """Create an official SysID hook for this model's two scalar sensors.

    The initial state must be explicit FULLPHYSICS [time, qpos, qvel], time 0.
    The caller pre-windows measured data to a fixed shared safe interval. The
    hook rejects unsafe support rather than accepting TimeSeries edge padding.
    It retains MuJoCo's measured-minus-predicted sign and weights by 1/sigma.
    Delay is shared and positive: prediction(t) = clean(t - delay).
    """
    initial = np.array(initial_state, dtype=np.float64, copy=True)
    if initial.shape != (3,) or not np.isfinite(initial).all() or initial[0] != 0:
        raise ValueError("initial_state must be finite FULLPHYSICS [0, qpos, qvel]")
    if set(sensor_sigmas) != {"qpos", "qvel"}:
        raise ValueError("sensor_sigmas must name exactly qpos and qvel")
    if not all(np.isfinite(s) and s > 0 for s in sensor_sigmas.values()):
        raise ValueError("sensor noise scales must be finite and positive")
    transform = sysid.SignalTransform(normalize=False)
    transform.set_sensor_weights({name: 1 / sigma for name, sigma in sensor_sigmas.items()})
    min_delay = max_delay = 0.0
    if delay_parameter is not None:
        if delay_parameter.size != 1:
            raise ValueError("shared delay must be a scalar Parameter")
        min_delay = float(delay_parameter.min_value[0])
        max_delay = float(delay_parameter.max_value[0])
        if not np.isfinite([min_delay, max_delay]).all() or not 0 <= min_delay < max_delay:
            raise ValueError("delay bounds must be finite, ordered, and nonnegative")
        transform.delay("*", delay_parameter)

    def aligned_hook(params, predicted, measured, model, return_pred_all, state=None):
        if (model.nq, model.nv, model.na, model.nsensordata) != (1, 1, 0, 2):
            raise ValueError("aligned hook supports only nq=nv=1, na=0, two sensors")
        if model.njnt != 1 or model.jnt_type[0] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError("aligned hook requires exactly one hinge joint")
        for expected_address, (name, expected_type) in enumerate(
            (("qpos", mujoco.mjtSensor.mjSENS_JOINTPOS), ("qvel", mujoco.mjtSensor.mjSENS_JOINTVEL))
        ):
            sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            if sensor_id < 0 or model.sensor_type[sensor_id] != expected_type:
                raise ValueError("aligned hook requires named jointpos/qvel sensors")
            if model.sensor_objid[sensor_id] != 0 or model.sensor_dim[sensor_id] != 1:
                raise ValueError("aligned sensors must observe the only hinge joint")
            if model.sensor_adr[sensor_id] != expected_address:
                raise ValueError("aligned sensor order must be qpos then qvel")
        states = np.asarray(state)
        if states.ndim != 2 or states.shape[1] != 3 or not np.isfinite(states).all():
            raise ValueError("state must contain finite FULLPHYSICS [time, qpos, qvel] rows")
        state_times = states[:, 0]
        if state_times.size == 0 or state_times[0] <= initial[0] or not np.all(np.diff(state_times) > 0):
            raise ValueError("post-step state timestamps must increase after initial time")
        grid = np.arange(states.shape[0] + 1, dtype=np.float64) * model.opt.timestep
        # Rollout adds dt repeatedly; datasets use the exact arange*dt grid.
        # Verify arithmetic agreement before canonicalising timestamps. This is
        # float accumulation correction, never a one-step sensor shift.
        time_tol = 1e-12 + 8 * np.finfo(np.float64).eps * states.shape[0] * max(1.0, float(grid[-1]))
        if not np.allclose(state_times, grid[1:], rtol=0, atol=time_tol):
            raise ValueError("post-step state timestamps disagree with the fixed simulation grid")
        clean = sysid.TimeSeries.from_names(
            grid,
            np.vstack((initial[1:], states[:, 1:3])), model, ["qpos", "qvel"],
        )
        if measured.signal_mapping != clean.signal_mapping:
            raise ValueError("observations must use the exact qpos/qvel sensor mapping")
        if measured.times.size == 0:
            raise ValueError("measured observation window must not be empty")
        # Verify support at both extreme delay bounds, independent of candidate.
        tol = 1e-12
        if measured.times[0] - max_delay < clean.times[0] - tol or measured.times[-1] - min_delay > clean.times[-1] + tol:
            raise ValueError("fixed measured window is unsafe for delay bounds; edge padding forbidden")
        if delay_parameter is not None:
            current_delay = float(params[delay_parameter.name].value[0])
            if not np.isfinite(current_delay) or not min_delay <= current_delay <= max_delay:
                raise ValueError("delay candidate is outside registered bounds")
        output = transform.apply(params, clean, measured, model, return_pred_all)
        if not np.array_equal(output[2].times, measured.times):
            raise ValueError("official delay window changed the fixed observation samples")
        return output

    return aligned_hook


def _physical_modifier_checks(spec: mujoco.MjSpec, model: mujoco.MjModel) -> dict:
    base_mass = float(model.body("link").mass[0])
    params = sysid.ParameterDict()
    mass = sysid.body_inertia_param(
        spec, model, "link", inertia_type=sysid.InertiaType.Mass,
        scale_rot_inertia=True, mass_bound_mult=np.array([0.5, 1.5]), param_name="mass",
    )
    params.add(mass)
    mass.update_from_vector(np.array([base_mass * 1.25]))
    changed = sysid.apply_param_modifiers(params, spec.copy())
    ratio_error = float(np.max(np.abs(changed.body("link").inertia / model.body("link").inertia - 1.25)))
    np.testing.assert_allclose(changed.body("link").mass, base_mass * 1.25, rtol=1e-14)
    np.testing.assert_allclose(changed.body("link").inertia, model.body("link").inertia * 1.25, rtol=1e-14)
    np.testing.assert_array_equal(changed.body("link").ipos, model.body("link").ipos)

    def damping_modifier(s, p):
        s.joint("hinge").damping = [float(p.value[0]), 0.0, 0.0]

    base_damping = float(model.dof_damping[0])
    frozen = sysid.Parameter("damping", base_damping + 0.1, 0.001, 0.5, frozen=True, modifier=damping_modifier)
    fixed_params = sysid.ParameterDict()
    fixed_params.add(frozen)
    fixed_model = sysid.apply_param_modifiers(fixed_params, spec.copy())
    np.testing.assert_array_equal(fixed_model.dof_damping, model.dof_damping)
    frozen.frozen = False
    free_model = sysid.apply_param_modifiers(fixed_params, spec.copy())
    np.testing.assert_allclose(free_model.dof_damping[0], base_damping + 0.1, rtol=1e-14)
    return {
        "mass_and_inertia_scaled_together": True,
        "inertia_ratio_max_abs_error": ratio_error,
        "com_unchanged": True,
        "frozen_physical_modifier_skipped": True,
        "fixed_physics_policy": "stamp into MjSpec; omit from free ParameterDict",
        "mjsjoint_damping_shape": list(np.asarray(spec.joint("hinge").damping).shape),
    }


def _pipeline_checks(spec: mujoco.MjSpec, model: mujoco.MjModel) -> dict:
    # Nonzero state and varying controls expose timing mistakes that equilibrium
    # fixtures conceal. This is a contract probe, not a study trajectory or fit.
    times = np.arange(201, dtype=np.float64) * model.opt.timestep
    controls = (0.45 * np.sin(np.arange(times.size) * 0.37))[:, None]
    q0, v0 = 0.17, 0.23
    direct = simulate(model, times, controls, initial_qpos=q0, initial_qvel=v0)
    control = sysid.TimeSeries.from_control_names(times, controls, model, ["torque"])
    initial = sysid.create_initial_state(model, np.array([q0]), np.array([v0]))
    official = sysid.sysid_rollout(model, mujoco.MjData(model), control, initial)[0]
    np.testing.assert_array_equal(official.sensordata.data, direct.sensordata[:-1])
    np.testing.assert_allclose(official.sensordata.times, times[1:], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(official.state.data[:, 1:], direct.sensordata[1:])
    safe = (times >= 0.060) & (times <= times[-1] - 0.060)
    sigmas = {"qpos": 0.002, "qvel": 0.020}
    measured = sysid.TimeSeries.from_names(times[safe], direct.sensordata[safe], model, ["qpos", "qvel"])
    sequence = sysid.ModelSequences("contract_probe", spec.copy(), "varying_control", initial, control, measured)
    params = sysid.ParameterDict()
    # Free parameter is held unchanged: no call to optimize occurs.
    params.add(sysid.body_inertia_param(spec, model, "link", sysid.InertiaType.Mass,
                                       scale_rot_inertia=True, param_name="mass"))
    hook = make_aligned_observation_hook(initial, sigmas)
    residual_fn = sysid.build_residual_fn(models_sequences=[sequence], modify_residual=hook, n_threads=1)
    residuals, _, _ = residual_fn(params.as_vector(), params)
    identity_error = float(np.max(np.abs(np.concatenate(residuals))))
    np.testing.assert_allclose(identity_error, 0.0, rtol=0, atol=1e-10)
    offsets = np.column_stack((np.sin(np.arange(safe.sum()) * 0.31),
                               np.cos(np.arange(safe.sum()) * 0.19)))
    errors = offsets * np.array([sigmas["qpos"], sigmas["qvel"]])
    noisy = sysid.TimeSeries.from_names(measured.times, measured.data + errors, model, ["qpos", "qvel"])
    noisy_sequence = sysid.ModelSequences("weighted_probe", spec.copy(), "known_offsets", initial, control, noisy)
    weighted_fn = sysid.build_residual_fn(models_sequences=[noisy_sequence], modify_residual=hook, n_threads=1)
    weighted_residuals, _, _ = weighted_fn(params.as_vector(), params)
    weight_error = float(np.max(np.abs(np.concatenate(weighted_residuals) - offsets.ravel())))
    np.testing.assert_allclose(np.concatenate(weighted_residuals), offsets.ravel(), rtol=0, atol=1e-10)
    normalizer_probe = np.array([[1.0, 2.0], [3.0, 5.0], [4.0, 9.0]])
    normalized = sysid.normalize_residual(np.ones_like(normalizer_probe), normalizer_probe)
    np.testing.assert_array_equal(normalized, 1 / (np.linalg.norm(normalizer_probe, axis=0) / np.sqrt(2)) * np.ones_like(normalizer_probe))
    return {
        "official_pipeline_executed_without_optimization": True,
        "probe_initial_qpos_rad": q0, "probe_initial_qvel_rad_s": v0,
        "probe_samples": int(times.size), "fixed_observation_samples": int(safe.sum()),
        "probe_duration_s": float(times[-1]),
        "probe_controls": {"formula": "0.45*sin(sample_index*0.37)", "shape": [201, 1], "terminal_control_unused": True},
        "probe_fixed_window_s": [0.060, float(times[-1] - 0.060)],
        "probe_sensor_sigmas": sigmas,
        "probe_weighted_offsets": {"qpos": "sigma_qpos*sin(window_sample_index*0.31)", "qvel": "sigma_qvel*cos(window_sample_index*0.19)"},
        "normalizer_fixture": normalizer_probe.tolist(),
        "upstream_sensor_timestamp_offset_s": float(model.opt.timestep),
        "upstream_accumulated_clock_max_abs_error_s": float(np.max(np.abs(official.state.data[:, 0] - times[1:]))),
        "upstream_sensors_equal_pre_step": True,
        "upstream_post_step_sensor_max_abs_error": float(np.max(np.abs(official.sensordata.data - direct.sensordata[1:]))),
        "aligned_identity_residual_max_abs": identity_error,
        "noise_weighted_residual_max_abs_error": weight_error,
        "residual_sign": "measured_minus_predicted",
        "default_normalizer": "column_l2_norm_divided_by_sqrt_2; not standard deviation",
        "hook": "modify_residual: explicit initial observation plus fresh FULLPHYSICS qpos/qvel; SignalTransform(normalize=False)",
        "timestamp_policy": "verify accumulated state times, then canonical arange*dt grid; no timestep shift",
        "timestamp_roundoff_tolerance_s": "1e-12 + 8*float64_epsilon*number_of_steps*max(1, horizon_s); rejects a timestep-sized shift",
        "n_rollout_threads": 1,
    }


def _delay_checks(model: mujoco.MjModel) -> dict:
    times = np.arange(501, dtype=np.float64) * 0.002
    data = np.column_stack((times, 2 * times + 1))
    clean = sysid.TimeSeries.from_names(times, data, model, ["qpos", "qvel"])
    unsafe = sysid.apply_delayed_ts_window(clean, clean, 0.0, 0.060)
    unsafe_count = int(np.count_nonzero(unsafe.times - 0.060 < clean.times[0]))
    if unsafe_count == 0:
        raise AssertionError("expected installed positive-delay window mismatch was not reproduced")
    mask = (times >= 0.060) & (times <= times[-1] - 0.060)
    safe = sysid.TimeSeries.from_names(times[mask], data[mask], model, ["qpos", "qvel"])
    windowed = sysid.apply_delayed_ts_window(safe, clean, 0.0, 0.060)
    np.testing.assert_array_equal(windowed.times, safe.times)
    max_error = 0.0
    for delay in (0.0, 0.024, 0.060):
        query = safe.times - delay
        assert query[0] >= clean.times[0] and query[-1] <= clean.times[-1]
        delayed = sysid.apply_resample_and_delay(clean, safe.times, delay)
        expected = np.column_stack((query, 2 * query + 1))
        np.testing.assert_allclose(delayed.data, expected, rtol=0, atol=1e-14)
        max_error = max(max_error, float(np.max(np.abs(delayed.data - expected))))
    return {
        "positive_delay_samples_t_minus_delay": True,
        "upstream_unwindowed_unsafe_samples_at_max_delay": unsafe_count,
        "fixed_safe_window_s": [0.060, 0.940],
        "delays_tested_s": [0.0, 0.024, 0.060],
        "safe_window_candidate_independent": True,
        "safe_query_support_verified": True,
        "monotone_fixture_max_abs_error": max_error,
        "fixture": {"timestep_s": 0.002, "samples": 501, "qpos": "t", "qvel": "2*t+1", "delay_bounds_s": [0.0, 0.060]},
        "time_series_outside_support_behavior": "constant endpoint padding; disallowed by project hook",
    }


def _finite_difference_checks() -> dict:
    lower, upper = np.array([[0.4], [0.001]]), np.array([[2.0], [0.5]])
    point = np.array([[0.4], [0.5]])
    matrix = np.array([[2.0, 3.0], [-1.0, 4.0], [0.2, -0.3]])
    seen = []

    def residual(x):
        seen.append(np.array(x, copy=True))
        if (x < lower).any() or (x > upper).any():
            raise AssertionError("official finite differences evaluated outside bounds")
        return matrix @ x

    jacobian, evaluations = minimize.jacobian_fd(
        residual, point, residual(point), np.float64(np.sqrt(np.finfo(np.float64).eps)), 0,
        bounds=[lower, upper],
    )
    np.testing.assert_allclose(jacobian, matrix, rtol=0, atol=5e-8)
    return {
        "official_batched_fd_bounds_respected": True,
        "analytic_jacobian_max_abs_error": float(np.max(np.abs(jacobian - matrix))),
        "reported_added_residual_evaluations": int(evaluations),
        "python_residual_calls": len(seen),
        "fixture": {"point": point.ravel().tolist(), "lower": lower.ravel().tolist(), "upper": upper.ravel().tolist(), "matrix": matrix.tolist(), "eps": float(np.sqrt(np.finfo(np.float64).eps))},
        "interpretation": "batched Jacobian probe only; no optimizer and no fitted physical parameter",
    }


def check_api(model_path: Path, physical_config: ModelConfig) -> dict:
    """Return strict-JSON-ready source fingerprints and executed contracts."""
    if mujoco.__version__ != "3.15.0":
        raise RuntimeError(f"registered MuJoCo is 3.15.0, installed {mujoco.__version__}")
    missing = [name for name in REQUIRED_SYMBOLS if not callable(getattr(sysid, name, None))]
    if missing:
        raise RuntimeError(f"required mujoco[sysid] symbols unavailable: {missing}")
    source_records = {name: _source_record(getattr(sysid, name)) for name in REQUIRED_SYMBOLS}
    for name, obj in (("ParameterDict.add", sysid.ParameterDict.add),
                      ("TimeSeries.from_names", sysid.TimeSeries.from_names),
                      ("TimeSeries.from_control_names", sysid.TimeSeries.from_control_names),
                      ("residual", sysid.residual),
                      ("SignalTransform.apply", sysid.SignalTransform.apply),
                      ("jacobian_fd", minimize.jacobian_fd)):
        source_records[name] = _source_record(obj)
    optimize_module = importlib.import_module("mujoco.sysid._src.optimize")
    wrapper = inspect.getsource(optimize_module._scipy_least_squares)
    native = inspect.getsource(optimize_module._mujoco_least_squares)
    builder = inspect.getsource(sysid.build_residual_fn)
    if 'kwargs.pop("max_iters", 200)' not in wrapper or "max_nfev=max_nfev" not in wrapper:
        raise RuntimeError("installed SciPy wrapper budget contract changed; inspect before use")
    if "signal_transform" not in builder or "signal_transform" in inspect.signature(sysid.residual).parameters:
        raise RuntimeError("installed residual-hook contract differs from inspected 3.15.0 release")
    source_records["scipy_backend_wrapper"] = _source_record(optimize_module._scipy_least_squares)
    spec = load_spec(model_path, physical_config)
    model = spec.compile()
    return {
        "schema_version": "mujoco-sysid-lab.api-check.v1",
        "mujoco_version": mujoco.__version__,
        "required_exports_present": True,
        "probe_physical_config": asdict(physical_config),
        "model_compilation": {"source": "load_spec: authored MJCF with exact physical config stamped", "model_sha256": sha256(Path(model_path).read_bytes()).hexdigest(), "state_spec": "mjSTATE_FULLPHYSICS"},
        "source_records": source_records,
        "physical_modifier_checks": _physical_modifier_checks(spec, model),
        "pipeline_checks": _pipeline_checks(spec, model),
        "delay_checks": _delay_checks(model),
        "finite_difference_checks": _finite_difference_checks(),
        "optimizer_inspection": {
            "planned_backend": "scipy_parallel_fd",
            "executed_estimation": False,
            "max_iters_semantics": "forwarded as scipy max_nfev; not iteration count",
            "max_nfev_direct_keyword": "do not pass: duplicates wrapper keyword",
            "diff_step_semantics": "scalar relative FD scale, multiplied by max(1, abs(x)); bounded batched mujoco.minimize.jacobian_fd",
            "scipy_diagnostics": ["success", "status", "message", "nfev", "njev", "cost", "optimality", "active_mask", "jac"],
            "iteration_count": {"value": None, "reason": "SciPy OptimizeResult does not expose a general nit field"},
            "native_backend_missing_termination_diagnostics": not all(f"{name}=" in native for name in ("success", "status", "nfev")),
            "supported_forwarded_options": ["method", "loss", "x_scale", "ftol", "xtol", "gtol", "callback"],
            "check_conditioning": "optional extra FD Jacobian warning; false by default",
            "verbose_semantics": "optimize(verbose=False) disables comparison log, but its argument is not forwarded; upstream SciPy wrapper still defaults to verbosity 2",
            "no_start_movement": "preserve preregistered starts; do not call move_off_bounds implicitly",
        },
        "report_inspection": {
            "executed_result_save_or_html_generation": False,
            "save_results_outputs": ["params_x_0.yaml", "params_x_hat.yaml", "results.pkl", "confidence.pkl", "<model_sequence_name>.xml"],
            "html_entrypoint": "default_report(..., save_path=Path(directory), generate_videos=False)",
            "html_output": "<directory>/report.html",
            "nominal_report_label": "measured observations; differs from project's biased nominal model",
            "confidence_intervals": "calculate_intervals available; not estimated by Milestone 1",
        },
        "mismatches": [
            "build_residual_fn docstring uses unsupported signal_transform; use modify_residual=hook",
            "frozen physical parameter modifiers are skipped; stamp fixed physics in MjSpec",
            "normalizer docstring says standard deviation; code is column L2 norm / sqrt(2)",
            "mass helper defaults to scale_rot_inertia=False; explicitly enable fixed-geometry scaling",
            "positive-delay window lower bound is unsafe without prespecified safe prewindow",
            "official rollout labels pre-step sensors with post-step times; explicit state observation hook required",
            "max_iters is SciPy max_nfev, despite iteration language in optimize docstring",
            "save_results writes raw outputs; HTML is generated separately by default_report",
        ],
    }
