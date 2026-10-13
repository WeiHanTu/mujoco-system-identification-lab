"""Finite fit diagnostics and the released toolbox's result/HTML writers.

MuJoCo 3.15 exports calculate_intervals, not compute_param_intervals. Its
intervals are local Student-t half widths using an estimated residual variance.
Official parameter-table errors use the starting reference, not hidden truth.
The project summary below is authoritative for truth errors and bound hits.
"""

from copy import copy, deepcopy
import json
from pathlib import Path
from typing import Any, Mapping

import mujoco
from mujoco import sysid
import numpy as np

from mujoco_sysid_lab.provenance import sha256_file


UNITS = {"mass_kg": "kg", "damping_nm_s_per_rad": "N*m*s/rad", "delay_s": "s"}
TERMINATION = {
    -2: "Stopped by callback.", -1: "Improper input parameters.",
    0: "Maximum function evaluation budget exceeded.",
    1: "gtol termination condition satisfied.",
    2: "ftol termination condition satisfied.",
    3: "xtol termination condition satisfied.",
    4: "Both ftol and xtol termination conditions satisfied.",
}


def _number(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real scalar")
    return result


def _flat(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "fiu" or array.ndim != 1 or not array.size or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a nonempty finite real vector")
    return array.astype(np.float64, copy=False)


def _diagnostic(result: Any, key: str) -> tuple[Any, str | None]:
    value = getattr(result, key, None)
    if value is None:
        return None, f"Official optimizer result does not expose {key}."
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None, f"Official optimizer returned non-finite {key}."
    return value, None


def _sensitivity(jacobian: Any, widths: np.ndarray, residual_count: int) -> dict:
    base = {
        "evaluation_point": "optimized parameter vector",
        "residual_basis": "registered noise-normalized training observations",
        "column_scaling": "parameter bound widths",
        "column_scale_values": widths.tolist(),
    }
    if jacobian is None:
        return {**base, "status": "unavailable", "reason": "Optimizer result has no Jacobian.",
                "singular_values": None, "rank": None, "condition_number": None}
    jac = np.asarray(jacobian)
    if jac.shape != (residual_count, widths.size) or jac.dtype.kind not in "fiu" or not np.isfinite(jac).all():
        return {**base, "status": "unavailable", "reason": "Jacobian is non-finite or has inconsistent shape.",
                "singular_values": None, "rank": None, "condition_number": None}
    try:
        with np.errstate(over="raise", invalid="raise"):
            scaled = jac * widths[None, :]
            singular_values = np.linalg.svd(scaled, compute_uv=False)
    except (FloatingPointError, np.linalg.LinAlgError):
        return {**base, "status": "unavailable", "reason": "Scaled Jacobian SVD failed.",
                "singular_values": None, "rank": None, "condition_number": None}
    if not np.isfinite(singular_values).all():
        raise ValueError("scaled Jacobian singular values must be finite")
    tolerance = float(max(scaled.shape) * np.finfo(np.float64).eps * singular_values[0])
    rank = int(np.count_nonzero(singular_values > tolerance))
    full_rank = rank == widths.size
    condition = float(singular_values[0] / singular_values[-1]) if full_rank else None
    if condition is not None and not np.isfinite(condition):
        condition = None
    return {
        **base, "status": "full_column_rank" if full_rank else "rank_deficient",
        "reason": None if full_rank else "Scaled Jacobian is rank deficient; no finite condition number reported.",
        "shape": list(jac.shape), "singular_values": singular_values.tolist(),
        "rank": rank, "rank_tolerance": tolerance, "condition_number": condition,
        "condition_number_reason": None if condition is not None else "A finite full-rank condition number is unavailable.",
    }


def _intervals(residual: np.ndarray, jacobian: Any, sensitivity: dict,
               estimates: np.ndarray, *, noiseless: bool, at_bound: bool,
               converged: Any) -> dict:
    base = {
        "method": "mujoco.sysid.calculate_intervals",
        "confidence_level": 0.95,
        "convention": "local Student-t half widths; estimated variance=sum(r**2)/(N-P)",
        "assumptions": [
            "Matched dynamics and observation model, near a suitable optimum.",
            "Independent Gaussian observation errors with registered relative noise scales.",
            "Local linear approximation and full-column-rank residual Jacobian.",
            "These are not five-seed robustness or real-robot confidence intervals.",
        ],
    }
    reason = None
    if noiseless:
        reason = "Noiseless numerical checks do not provide statistical measurement-noise confidence intervals."
    elif sensitivity["status"] != "full_column_rank":
        reason = "A finite full-column-rank Jacobian is required."
    elif residual.size <= estimates.size:
        reason = "Positive residual degrees of freedom are required."
    if reason:
        return {**base, "status": "unavailable", "reason": reason,
                "covariance": None, "half_widths": None, "intervals": None}
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            covariance, half_widths = sysid.calculate_intervals([residual], np.asarray(jacobian), alpha=0.05)
        if not np.isfinite(covariance).all() or not np.isfinite(half_widths).all():
            raise ValueError("official interval calculation returned non-finite values")
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        return {**base, "status": "unavailable", "reason": str(exc),
                "covariance": None, "half_widths": None, "intervals": None}
    caveats = []
    if at_bound:
        caveats.append("A parameter hits a registered bound; symmetric local intervals are unreliable.")
    if converged is not True:
        caveats.append("Optimizer convergence is unverified; local intervals are unreliable.")
    return {**base, "status": "available_with_caveats" if caveats else "available",
            "reason": None, "caveats": caveats, "covariance": covariance.tolist(),
            "half_widths": half_widths.tolist(),
            "intervals": np.column_stack((estimates - half_widths, estimates + half_widths)).tolist()}


def summarize_fit(problem: Any, execution: Any, truth_values: Mapping[str, float], *,
                  noiseless: bool = False, bound_hit_fraction: float = 0.001) -> dict:
    """Summarize one retained fit; truth is evaluation-only and never refitted."""
    fraction = _number(bound_hit_fraction, "bound_hit_fraction")
    if not 0 <= fraction < 0.5:
        raise ValueError("bound_hit_fraction must lie in [0,0.5)")
    initial_r = _flat(execution.initial_residual, "initial_residual")
    final_r = _flat(execution.final_residual, "final_residual")
    if initial_r.shape != final_r.shape:
        raise ValueError("initial and final residuals must have the same fixed sample shape")
    params, optimized = problem.params, execution.optimized_params
    names = params.get_non_frozen_parameter_names()
    if names != optimized.get_non_frozen_parameter_names() or not names:
        raise ValueError("initial and optimized free parameter names must agree and be nonempty")
    initial = _flat(params.as_vector(), "initial parameters")
    estimates = _flat(optimized.as_vector(), "optimized parameters")
    lower, upper = (np.asarray(array, dtype=np.float64) for array in params.get_bounds())
    if (initial.shape != estimates.shape or len(names) != initial.size
            or lower.shape != initial.shape or upper.shape != initial.shape
            or not np.isfinite(lower).all() or not np.isfinite(upper).all()
            or np.any(lower >= upper) or np.any(estimates < lower) or np.any(estimates > upper)):
        raise ValueError("free parameter bounds/vectors are inconsistent or infeasible")
    rows = []
    for index, name in enumerate(names):
        truth = _number(truth_values[name], f"truth_values.{name}") if name in truth_values else None
        if truth is None:
            raise ValueError(f"missing evaluation truth for {name}")
        absolute = abs(float(estimates[index]) - truth)
        lo_hit = bool(estimates[index] - lower[index] <= fraction * (upper[index] - lower[index]))
        hi_hit = bool(upper[index] - estimates[index] <= fraction * (upper[index] - lower[index]))
        rows.append({
            "name": name, "unit": UNITS.get(name, "unspecified"),
            "initial": float(initial[index]), "estimate": float(estimates[index]),
            "truth": truth, "lower_bound": float(lower[index]), "upper_bound": float(upper[index]),
            "absolute_error": absolute, "relative_error": absolute / abs(truth) if truth != 0 else None,
            "relative_error_reason": None if truth != 0 else "Relative error is undefined for zero truth.",
            "bound_hit": lo_hit or hi_hit, "lower_bound_hit": lo_hit, "upper_bound_hit": hi_hit,
            "bound_hit_fraction": fraction, "error_reference": "evaluation-only hidden physical truth",
        })
    result = execution.optimize_result
    diagnostics, unavailable = {}, {}
    for key in ("success", "status", "message", "nfev", "njev", "nit", "cost", "optimality"):
        diagnostics[key], reason = _diagnostic(result, key)
        if reason:
            unavailable[key] = reason
    diagnostics["unavailable_reasons"] = unavailable
    diagnostics["backend"] = "official_sysid_scipy_parallel_fd"
    diagnostics["termination_meaning"] = TERMINATION.get(diagnostics["status"], "Unavailable or unrecognized SciPy termination status.")
    diagnostics["callback_events"] = int(execution.callback_events)
    diagnostics["iteration_count_basis"] = "nit only if upstream exposes it; callback_events counted separately"
    jac = getattr(result, "jac", None)
    sensitivity = _sensitivity(jac, upper - lower, final_r.size)
    with np.errstate(over="raise", invalid="raise"):
        initial_objective = _number(0.5 * np.dot(initial_r, initial_r), "initial objective")
        final_objective = _number(0.5 * np.dot(final_r, final_r), "final objective")
    runtime = _number(execution.runtime_seconds, "runtime_seconds")
    if runtime < 0:
        raise ValueError("runtime_seconds must be nonnegative")
    summary = {
        "schema_version": "mujoco-sysid-lab.fit-summary.v1",
        "parameter_rows": rows,
        "objective": {"initial": initial_objective, "final": final_objective,
                      "definition": "0.5*sum(noise_normalized_residual**2)",
                      "residual_count": int(final_r.size),
                      "initial_mean_squared_normalized_residual": 2 * initial_objective / initial_r.size,
                      "final_mean_squared_normalized_residual": 2 * final_objective / final_r.size,
                      "native_final_cost": diagnostics["cost"],
                      "native_cost_reason": unavailable.get("cost"),
                      "native_cost_definition": "SciPy linear least-squares cost=0.5*sum(residual**2)"},
        "optimizer": diagnostics, "runtime_seconds": runtime,
        "runtime_basis": "Initial residual, official optimization, and final residual evaluations; excludes later save/report calls.",
        "call_counts": deepcopy(execution.call_counts),
        "call_count_basis": "Captured initial/optimization/final phases; excludes subsequent save/report evaluations.",
        "objective_trace": deepcopy(list(execution.objective_trace)),
        "sensitivity": sensitivity,
        "confidence_intervals": _intervals(final_r, jac, sensitivity, estimates, noiseless=noiseless,
                                           at_bound=any(row["bound_hit"] for row in rows),
                                           converged=diagnostics["success"]),
        "noiseless": bool(noiseless),
    }
    json.dumps(summary, allow_nan=False)
    return summary


def _copy_build_model(params: Any, spec: mujoco.MjSpec) -> mujoco.MjModel:
    return sysid.apply_param_modifiers(params, spec.copy())


def save_official_fit(directory: Path, problem: Any, execution: Any, *,
                      title: str = "Synthetic oracle-assisted SysID", noiseless: bool = False) -> dict:
    """Save official raw results, correctly stamped XML, and official HTML.

    The registered problem spec and captured optimization counts are preserved.
    Copied report groups receive fitted modifiers explicitly because the
    residual's copy-build hook intentionally leaves the original spec pristine.
    """
    from mujoco.sysid.report.sections.row import RowSection

    output = Path(directory)
    if output.exists():
        raise ValueError("Output already exists; choose a new run directory to preserve evidence")
    original_xml = problem.spec.to_xml()
    before = problem.residual_fn.snapshot_counts()
    groups = []
    for original_group in problem.sequences:
        group = copy(original_group)
        group.spec = original_group.spec.copy()
        sysid.apply_param_modifiers_spec(execution.optimized_params, group.spec)
        # MjSpec.to_file uses the compiled snapshot. Compile after changing
        # physics; otherwise 3.15 can serialize the copied initial values.
        group.spec.compile()
        groups.append(group)
    output.mkdir(parents=True, exist_ok=False)
    sysid.save_results(output, groups, problem.params, execution.optimized_params,
                       execution.optimize_result, problem.residual_fn)
    verification = {}
    for group in groups:
        path = output / f"{group.name}.xml"
        saved = mujoco.MjModel.from_xml_path(str(path))
        expected = group.spec.compile()
        for field in ("body_mass", "body_inertia", "body_ipos", "dof_damping", "dof_dampingpoly"):
            np.testing.assert_allclose(getattr(saved, field), getattr(expected, field), rtol=1e-12, atol=1e-12)
        verification[group.name] = {"saved_xml_matches_optimized_mass_inertia_damping": True,
                                    "path": path.name, "sha256": sha256_file(path)}
    rb = sysid.default_report(groups, problem.params, execution.optimized_params,
                              problem.residual_fn, execution.optimize_result,
                              title=title, save_path=None, build_model=_copy_build_model,
                              generate_videos=False)
    interval_note = (
        "This noiseless check is numerical verification; its interval numbers are not statistical confidence."
        if noiseless else
        "Confidence intervals are local linear Student-t approximations for independent observation noise; they are unreliable with model mismatch, bounds, or weak sensitivity."
    )
    rb.add_section(RowSection(
        title="Project interpretation and claim boundaries", sections=[],
        anchor="project-interpretation", collapsible=False,
        description=(
            "Synthetic oracle-assisted single-parameter study; no real robot, HIL, or sim-to-real validation. "
            "The official observation legend 'nominal' means measured signals, while this project's nominal model means biased physics. "
            "The official parameter error table compares against the starting reference, not hidden ground truth. "
            "Project fit summary truth errors and bound hits (0.001 times bound width) are authoritative; upstream bound styling uses a different rule. "
            "Optimization trace entries include initial, callback, and final evaluations; the upstream Iteration label does not establish an iteration count. "
            + interval_note
        ),
    ))
    rb.save(output / "report.html")
    if problem.spec.to_xml() != original_xml:
        raise RuntimeError("official saving/reporting mutated the registered problem spec")
    after = problem.residual_fn.snapshot_counts()
    before_total, after_total = before.get("total", {}), after.get("total", {})
    reporting_counts = {key: int(after_total[key]) - int(before_total.get(key, 0)) for key in after_total}
    files = {path.name: {"path": path.name, "sha256": sha256_file(path)}
             for path in sorted(output.iterdir()) if path.is_file()}
    return {"status": "passed", "files": files, "report": files["report.html"],
            "verified_xml": verification, "original_spec_unchanged": True,
            "reporting_call_counts": reporting_counts,
            "reporting_count_basis": "Additional residual evaluations by official save_results/default_report; excluded from optimizer snapshots.",
            "html_renderer": "mujoco.sysid.default_report and public ReportBuilder.add_section/save",
            "videos_generated": False}
