# Project interfaces

This is a repository-driven experiment, rather than a general robot-identification library. The supported entry points are the scripts in `scripts/`; the small Python package separates validated inputs, fitting, and post-selection evaluation.

## Estimation boundary

| Interface | Inputs and output | Contract |
|---|---|---|
| `load_study_config(path)` | Frozen JSON → `StudyConfig` | Contains nominal physics, bounds, known constants, and a truth-file pointer; no true physical values. Unknown fields, invalid bounds, non-finite numbers, or changed experiment structure raise `ValueError`. |
| `load_training_sequences(manifest)` | Core dataset → `TrainingDataset` | Opens only the two complete training NPZs and their five permitted arrays. No truth metadata, latent signals, or held-out arrays enter the returned object. |
| `load_scenario_training(manifest)` | Extended scenario dataset → `TrainingDataset` | Same restricted input view; supports the rich pair or the separate weak-step trajectory. Full scenario loading belongs to evaluation. |
| `build_problem(training, model_path, physical, study, free_parameters, start, *, training_ids=...)` | Restricted inputs → `FitProblem` | Free names are `mass_kg`, `damping_nm_s_per_rad`, and/or `delay_s`. Start keys must exactly match; finite values must lie within bounds. Only the rich training pair or `weak_step` is allowed. |
| `FitProblem.run()` | Prepared problem → `FitExecution` | Calls official `sysid.optimize`; captures native results, immutable residuals, callback history, runtime, and actual rollout-call counts. Native convergence status is retained. |

`physical` stamps fixed quantities into an `MjSpec`. Mass changes scale rotational inertia consistently. For delay-only fitting, fixed physics must come from the same seed's sealed training-selected B fit. The A1/A2 checks intentionally use a true nuisance parameter and are explicitly oracle-assisted.

A fitting example, after replacing the placeholder with a generated **core** dataset manifest:

```python
from pathlib import Path
from mujoco_sysid_lab.dataset import load_training_sequences
from mujoco_sysid_lab.experiment_config import load_study_config
from mujoco_sysid_lab.identification import build_problem

study = load_study_config(Path("configs/evidence_v1.json"))
training = load_training_sequences(Path("artifacts/<generated-run>/dataset/manifest.json"))
problem = build_problem(
    training, Path("model/pendulum.xml"), study.nominal, study,
    ("mass_kg", "damping_nm_s_per_rad"),
    {"mass_kg": 0.80, "damping_nm_s_per_rad": 0.04},
)
fit = problem.run()
```

This example supplies no physical truth or held-out trajectory. Use the full runner to preserve the complete reporting/provenance contract.

## Array and timing contract

| Array | Shape | Units / meaning |
|---|---|---|
| `timestamps` | `(N,)` | Seconds, strictly increasing uniform grid beginning at zero. |
| `controls` | `(N, 1)` | Motor control units; known gain is 1 N·m/unit. `u[k]` drives the next interval; the final control is unused. |
| `measured_qpos` | `(N, 1)` | Unwrapped radians. Zero is downward. |
| `measured_qvel` | `(N, 1)` | Radians/second. |
| `initial_state` | `(3,)` | Explicit `mjSTATE_FULLPHYSICS`: `[time, qpos, qvel]`; known equilibrium reset, not fitted noisy observations. |

All numeric arrays must be finite. The model contract is `nq=nv=nu=1`, `na=0`, with exactly the named joint-position and joint-velocity sensors. The adapter does not support arbitrary sensors or robots.

The registered 2 ms timestep and common observation window `[0.06, T-0.06]` apply to every fitting candidate and baseline. A positive shared delay means `measured(t) = clean(t-delay) + noise(t)`. Generation includes explicit equilibrium prehistory. Fitting pre-windows the measurements, validates support, and uses the official delay resampler on the exact measured grid. No padding or extrapolation is allowed.

The per-observation residual is `(measured - predicted) / sigma`, with `sigma_q=0.002 rad` and `sigma_v=0.02 rad/s`. The linear least-squares objective is `0.5 * sum(residual**2)`; amplitude normalization is disabled. Parameter vectors follow `free_parameters` order. Residuals are flattened across complete training sequences; callers should use named signals rather than assume a general sensor ordering.

## Evaluation and reporting

`evaluate_model(scenario, physical, model_path, study, delay_s=0)` is an **after-selection** interface. It independently rolls out each complete control from the known initial state and returns strict JSON containing training/held-out aggregate metrics, per-trajectory metrics, secondary latent-state diagnostics, and the generator-aware clean-observation reference. The fitting-family oracle uses true physics and delay but omits the mismatch force.

`evaluate_trajectory(...)` accepts the common timestamps, measured/predicted `(N,1)` position/velocity arrays, positive noise scales, and an optional common window. It returns position/velocity RMSE, noise-normalized RMSE, linear-method p95 absolute errors, and sample/window metadata. Split aggregation gives each trajectory equal weight. A pooled p95 is not invented from per-trajectory percentiles.

`relative_improvement(identified, nominal)` returns `1 - identified_rmse/nominal_rmse`. A zero denominator produces null plus a reason. Invalid shapes, timestamps, support, parameter ranges, and NaN/Inf raise `ValueError`; the runner retains failure records. SciPy iteration count is null when unsupported; callback events and native function/Jacobian evaluations are distinct.

Public result schemas are versioned. [overview.json](results/overview.json) is the compact entry point; [summary.json](results/summary.json) retains every fit's diagnostics and references. Confidence intervals are local linear approximations and do not establish general identifiability or true-physics coverage under model mismatch.

## Installed API corrections

The pinned package's behavior was inspected before adapting it: `modify_residual` replaces a stale example keyword; frozen physical modifiers are skipped; inertia scaling is opt-in; amplitude normalization differs from its docstring; rollout sensor labels are offset; and automatic delay cropping can drop an endpoint. The supported adapter is narrowly scoped to this model. Official XML rounds physical fields; native arrays and MuJoCo MJB files preserve full precision. [spec.md](../spec.md) records sources, tests, and exact conventions.
