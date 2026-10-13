# Proposed experimental specification v0.1

Status: user approved v0.1 on 2026-10-12; Milestone 1 installed-API verification is complete. Physical recovery and held-out targets below are registered proposals, not measured identification results. Serialize the full evidence specification into versioned JSON before the first identification evidence run. No outcome-driven revision is allowed.

## Environment and dependencies

Initial planning inspection on 2026-10-12, America/Los_Angeles (before the Milestone 1 installation):

- Workspace was empty; no on-disk ancestor/workspace `AGENTS.md` was found. The user supplied the directness instruction in chat.
- macOS 26.6.2, Darwin 25.6.0, ARM64; Apple M4 Pro, 12 CPU cores (8 performance, 4 efficiency), 24 GB memory.
- Default Python: 3.14.7. Proposed experiment interpreter already available: CPython 3.12.5 at `/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12`.
- MuJoCo was absent in both inspected interpreters. No dependencies were installed and no import/simulation success is claimed.
- Available tooling: uv 0.12.10 and Git 2.39.5. Project is a separate directory/repository under the current workspace.

Proposed exact direct pins:

| Dependency | Version | Purpose |
|---|---|---|
| Python | 3.12.5 | Existing local CPU interpreter; CI uses the same version |
| `mujoco[sysid]` | 3.15.0 | Released simulator and official SysID toolbox |
| `numpy` | 2.3.3 | Arrays, deterministic noise, metrics |
| `scipy` | 1.16.2 | SysID dependency and diagnostic numerical utilities |
| `matplotlib` | 3.10.6 | Static evidence plots with Agg backend |
| `pytest` | 8.4.2 | Development/test dependency |
| `setuptools` | 80.9.0 | Pinned project build backend |

These supporting versions are conservative released pins, not claims that they are the newest. PyPI release pages were checked. Milestone 1 resolved and installed 36 locked packages; required imports and deterministic residual/simulation contracts pass. Optimization and official HTML generation remain later milestones. Use JSON configs and stdlib parsing; no additional configuration framework. Committed uv.lock covers transitive runtime SysID extras and wheel hashes; setuptools==80.9.0 is pinned in build-system requirements. A future resolver conflict must be reported rather than silently changing pins.

Official release metadata confirms the `sysid` extra and report template packaging. Planning-stage read-only inspection of the released CPython 3.12 macOS ARM64 wheel confirmed required export declarations and important sources matching tag 3.15.0. The wheel SHA256 matched PyPI: `18d1b33660c6431ef05398fc576ff1e073ea4119c6e6836f8a113e393d915308`. That inspection did not import/install code; subsequent Milestone 1 installation verified the actual imports/signatures and contracts below:

- `Parameter`, `ParameterDict`, `TimeSeries`, `create_initial_state`, `ModelSequences`, `build_residual_fn`, `optimize`, `save_results`, and `default_report` must be present and callable.
- Import alone is insufficient: the release initializer can catch a missing optional dependency and leave exports unavailable.
- A tagged residual example references `signal_transform`, while the actual residual builder uses `modify_residual`. Do not copy that stale keyword.
- A frozen-parameter example/docstring disagrees with the modifier path, which skips frozen physical modifiers. Stamp fixed physical values into the spec and omit them from free physical parameters.
- The default residual includes signal-amplitude normalization. Installed 3.15.0 code actually divides by per-sensor L2 norm / sqrt(2), despite a docstring describing standard deviation. Disable it and explicitly apply known-noise weighting; otherwise the intended objective changes.
- The mass helper defaults to not scaling rotational inertia. Set consistent mass/inertia behavior explicitly and verify the compiled model.
- The installed positive-delay window probe reproduces 30 unsupported leading samples at maximum registered delay if data is not pre-windowed. The fixed safe pre-window is tested before delay identification.
- Installed runtime probes confirm a separate one-timestep sensor/state labeling offset for this jointpos/jointvel pair: official rollout sensors precede integration, but SysID arrays2traj labels them with post-step state times. Use the supported modify_residual hook to reconstruct these exact two sensors from FULLPHYSICS post-step states, prepend the explicit initial observation, and validate against direct fresh-sensor simulation. This narrow adapter is limited to nq=nv=1, na=0, and the named jointpos/jointvel sensor layout. Never silently fit the offset as measurement delay.
- Rollout accumulates floating-point time whereas the data grid uses arange*dt. Verify state times against that grid with arithmetic tolerance `1e-12 + 8*float64_eps*nsteps*max(1,horizon_s)`, then use canonical grid labels for the fresh observations. The eight-second timing test observed 6.59e-13 s clock accumulation error and preserved both fixed-window boundary samples. This arithmetic correction does not alter the 0.002 s scientific delay-recovery threshold.
- Native `optimizer="mujoco"` does not provide success/status/iteration/evaluation fields. Use the official toolbox's `scipy_parallel_fd` backend for richer diagnostics; do not fabricate unavailable fields or write a separate optimizer.
- The official optimizer wrapper calls its budget `max_iters` but forwards it as SciPy `max_nfev`. Passing `max_nfev` directly duplicates the keyword. Use the verified wrapper argument and distinguish this evaluation budget from an actual iteration count.
- `save_results` saves raw parameters/results/confidence/XML, not HTML. Generate HTML separately with `default_report`, a Path output directory, and videos disabled. Its upstream observation label "nominal" denotes measured signals; explain that it differs from this project's biased nominal model.

Primary sources:

- [MuJoCo 3.15.0 release](https://pypi.org/project/mujoco/3.15.0/)
- [Tagged package configuration](https://github.com/google-deepmind/mujoco/blob/3.15.0/python/pyproject.toml)
- [Tagged SysID exports](https://github.com/google-deepmind/mujoco/blob/3.15.0/python/mujoco/sysid/__init__.py)
- [Tagged residual builder](https://github.com/google-deepmind/mujoco/blob/3.15.0/python/mujoco/sysid/_src/residual.py)
- [Tagged signal modifiers](https://github.com/google-deepmind/mujoco/blob/3.15.0/python/mujoco/sysid/_src/signal_modifier.py)
- [Tagged optimizer](https://github.com/google-deepmind/mujoco/blob/3.15.0/python/mujoco/sysid/_src/optimize.py), [result saving](https://github.com/google-deepmind/mujoco/blob/3.15.0/python/mujoco/sysid/_src/io.py), [official HTML reporting](https://github.com/google-deepmind/mujoco/blob/3.15.0/python/mujoco/sysid/report/defaults.py)
- [NumPy pin](https://pypi.org/project/numpy/2.3.3/), [SciPy pin](https://pypi.org/project/scipy/1.16.2/), [Matplotlib pin](https://pypi.org/project/matplotlib/3.10.6/), [pytest pin](https://pypi.org/project/pytest/8.4.2/)
- [Build backend pin](https://pypi.org/project/setuptools/80.9.0/)

## Mechanical contract

- One hinge about local y; q = 0 is the vertically downward equilibrium. No joint limits, contacts, or ground collision in the core model.
- Uniform rigid cylinder: length L = 0.50 m, radius r = 0.015 m, COM distance L/2 = 0.25 m below the hinge. Geometry and COM are known/fixed.
- Explicit body inertia at COM: Ixx = Iyy = m(L²/12 + r²/4), Izz = mr²/2. All components scale with m. The effective hinge inertia includes the parallel-axis term; do not independently optimize inertia.
- Gravity = (0, 0, -9.81) m/s². Fixed timestep = sensor sample interval = 0.002 s. Proposed integrator: `implicitfast` for both target and fitting models.
- Motor control is torque with known gain/gear = 1.0 N*m per control unit. No activation dynamics, saturation, or fitted actuator gain. Control maximum absolute magnitude = 1.5 N*m; reject an excitation config exceeding it rather than silently clipping.
- Joint armature and dry friction are zero in the matched model. Damping units: N*m*s/rad.
- Named hinge position/velocity sensors. qpos is an unwrapped angle in radians; qvel is rad/s. No renderer is required for dynamics or CI.
- Exact initial state: qpos = 0, qvel = 0, time = 0, plus all required integration/activation state. No noisy initial-state estimation or per-window state refitting.

| Quantity | Hidden truth | Nominal | Allowed estimation bounds |
|---|---:|---:|---:|
| Mass (kg) | 1.20 | 0.80 | [0.40, 2.00] |
| Damping (N*m*s/rad) | 0.15 | 0.04 | [0.001, 0.50] |
| Shared sensor delay (s) | 0 in matched; 0.024 in delay study | 0 | [0, 0.060] |
| Known motor gain | 1.0 | 1.0 | Fixed |

Absolute mass recovery is conditional on the known gain, COM, geometry, and other dynamics. Fitting free mass/inertia/gain together would create scale ambiguity; that is outside the core claim.

## Data and timing contract

Each rich-training/held-out trajectory lasts 8.0 s, including a 0.10 s zero-control prefix. Controls are piecewise constant on simulation intervals. u[k] drives [t[k], t[k+1]); clean sensor values correspond to the state at t[k]. Verify MuJoCo step/forward and official rollout alignment explicitly rather than assuming sensor outputs are aligned after integration.

Training trajectories:

1. `train_multisine`: frequencies [0.25, 0.65, 1.15, 1.80] Hz, raw amplitudes [0.20, 0.30, 0.25, 0.15] N*m, phases [0, 0.7, 1.3, 2.1] rad; rescale the active segment to RMS torque 0.55 N*m.
2. `train_prbs`: independent seeded piecewise-constant choices from [-1, -0.6, 0, 0.6, 1], 0.25 s blocks, rescale the active segment to RMS torque 0.55 N*m. Excitation seed = 314159.

Held-out trajectories, excluded from estimation and start selection:

1. `heldout_chirp`: linear frequency sweep 0.20 to 2.00 Hz over the active segment, amplitude 0.80 N*m, with a 0.10 s cosine onset envelope.
2. `heldout_pulse`: torque 0 except +1.25 N*m on [0.50, 1.20), -0.90 on [2.50, 3.30), and +0.95 on [5.00, 5.45) s. These amplitudes differ from the training control construction.

Weak-training ablation: one 16.0 s step trajectory, control 0 for the first 0.10 s and +0.55 N*m afterwards. It matches nominal total training duration and active RMS torque, but differs in reset count and transient structure. Record actual sampled energy and RMS; interpretation must acknowledge those differences. A step may be sufficient to identify both parameters; its failure is not an acceptance criterion.

Noise seeds = [101, 102, 103, 104, 105]. Independent zero-mean Gaussian additive observation noise with sigma_q = 0.002 rad and sigma_v = 0.020 rad/s. No process noise. Use PCG64 with SeedSequence([noise_seed, noise_group_id, trajectory_id, sensor_id]), never Python's randomized hash. Trajectory IDs: multisine = 0, PRBS = 1, chirp = 2, pulse = 3, weak step = 4; sensor IDs: qpos = 0, qvel = 1. Noise group 0 is shared by matched, delayed, and mismatch scenarios, which reuse identical standard-normal innovations for paired trajectories. Weak training uses group 1; its held-out evaluations reuse group-0 held-out innovations. Excitation randomness is separate, using SeedSequence([314159, 100, 1]) for PRBS controls. Save the scheme, group/trajectory/sensor keys, and innovation hashes. Noiseless checks disable observation noise without changing the positive residual-weight scales.

For positive sensor lag tau, measured y(t) = clean sensor(t - tau) + noise(t). Generate a clean zero-control equilibrium prehistory from -0.060 to 0 s, followed by the trajectory's 0.10 s zero-control prefix. Export initial state and timestamps from 0 s, and record/save the prehistory separately; this supplies valid sensor history even at t = 0 without edge padding. Apply delay to clean signals first, then add noise. Retain complete data; use a fixed common residual/evaluation window [0.060, T - 0.060] s for every candidate and all three baselines, including zero delay. Interpolation of predictions must stay inside their time support, with no extrapolation, candidate-dependent cropping, angle wrapping, or hidden one-step shift. Verify sign using a monotone signal fixture and dynamics timing test.

Phase-lag diagnostic: separately for qpos/qvel, compute demeaned normalized cross-correlation between candidate observation prediction p(t) and measured y(t + lag), using fixed t in [0.120, T - 0.120] s for every lag. Search lag from -0.060 to +0.060 s on the 0.002 s sensor grid. Positive lag means measurements lag the candidate prediction. Choose the maximum correlation; ties within 1e-12 prefer smallest absolute lag, then negative lag. Report signed lag, absolute lag, and peak correlation for each sensor, plus median absolute lag across valid sensors. Return null/reason if either signal standard deviation is below that sensor's noise sigma or peak correlation is below 0.20. This is a diagnostic, not the delay estimator or an uncertainty interval. Compare absolute lag before/after the identified measurement transform without adjusting the correlation rules after seeing results.

Dataset format: `dataset_schema_version = 1`, per-trajectory NPZ plus JSON manifest. No pickle/object arrays. Arrays: timestamps [N], controls [N,1], measured_qpos [N,1], measured_qvel [N,1], clean latent qpos/qvel for evaluation only, and explicit initial state. The final control sample is documented as an unused terminal value. Manifest includes IDs/split/scenario, units, duration/dt, exact excitation/noise config, hidden truth, model/config hashes, seed keys, version/environment/command/source provenance, and array hashes. Require strictly increasing uniform timestamps, matching shapes, valid train/held-out IDs, and finite values.

Keep ground truth in a separate config/metadata path. The joint-estimator loader exposes only training controls, measured signals, known initialization, and nominal/bound config; it must not receive clean targets, truth, or held-out trajectories. The A1/A2 unit checks are explicitly oracle-assisted: only the nuisance physical parameter is stamped to truth by the test harness. The free parameter still starts at nominal. This exception does not apply to B, delay fits, weak excitation, or mismatch. "Hidden" means hidden from core estimation, not secret from readers of this synthetic study.

## Official SysID and objective contract

Load authored MJCF with MjSpec and compile it. Create bounded Parameters in ParameterDict with supported spec modifiers for fixed-geometry mass/inertia and joint damping. Build named control/measured TimeSeries, explicit initial states, and ModelSequences with complete training trajectories. Build the official residual with build_residual_fn; run sysid.optimize using the toolbox's official `scipy_parallel_fd` backend, which retains MuJoCo batched finite-difference rollouts. Save official results and generate the official HTML report through the installed supported report API. This is a supported official backend, not a separately implemented optimizer.

Use the supported measurement-transform/residual hook to align jointpos/jointvel with the returned integration states, disable amplitude normalization, and weight each sensor residual by 1/sigma, not 1/sigma². Prefer official SignalTransform helpers after alignment and pre-windowing the measured series. Any small adapter must be limited to verified API/timing plumbing, documented with signatures and tests; it cannot replace optimization or create a second scientific objective.

Residual r[k,s] = (measured[k,s] - prediction[k,s]) / sigma_s, preserving the official weighted_diff sign. Installed inspection corrected the planning-stage sign; the least-squares objective is unchanged. Intended objective is 0.5 * sum(r²); record that recomputed value and the optimizer's native objective/convention. Rich trajectories have equal length and sensor counts. Compare scenarios using mean normalized residual squared, not raw sum alone.

Optimizer proposal: official `scipy_parallel_fd` least-squares backend; method = "trf", linear loss, wrapper max_iters = 200 (mapped to SciPy max_nfev), ftol = xtol = gtol = 1e-8, x_scale equal to free-parameter bound widths, and diff_step = 1e-5. The wrapper owns jac/bounds. Its finite-difference step is eps * max(1, abs(parameter)), with sign toward the bounds midpoint and candidate clipping; it is not SciPy's relative-x diff_step convention. Verify installed behavior and nonzero effective steps at bounds, and serialize every option before evidence. Fixed deterministic thread settings: one rollout worker, one BLAS/OpenMP thread. Store function/Jacobian evaluations, backend success/status/message, actual termination meaning, and perf_counter wall time. Record history and iteration events with the supported SciPy callback, labeling the count as callback events rather than inventing an upstream nit field. If unavailable, store null with a reason. Count residual calls and logical perturbed rollouts separately because backend nfev excludes callable-Jacobian work and reporting evaluations. Any diagnostic absent from this backend is explicitly unavailable. Bitwise determinism is tested within the recorded environment; do not promise bitwise equality across operating systems or architectures.

Starts for physical joint fitting:

- S0 nominal: mass 0.80, damping 0.04.
- S1: mass 1.70, damping 0.30.
- S2: mass 0.50, damping 0.40.

Joint delay starts paired with S0/S1/S2: 0.002, 0.050, 0.010 s. Use an interior start for optimized delay; retain zero as a separately evaluated baseline. Delay-only fits start at 0.002 s. At each seed, choose the start with lowest finite training objective only; include non-convergence flags and report every start. Do not filter a finite non-converged fit merely to improve headlines. All-invalid seeds remain explicit failures.

## Registered evidence matrix: 82 optimization fits

| Experiment | Data | Free parameters | Noise seeds | Starts | Fits |
|---|---|---|---:|---:|---:|
| A1 oracle-assisted mass sanity | Matched rich training | Mass; truth damping stamped in spec | 101 | Nominal mass | 1 |
| A2 oracle-assisted damping sanity | Matched rich training | Damping; truth mass stamped in spec | 101 | Nominal damping | 1 |
| B matched joint | Matched rich training | Mass + damping | 5 | 3 | 15 |
| Weak excitation | Matched 16 s step | Mass + damping | 5 | 3 | 15 |
| Delay excluded | Delayed rich training | Mass + damping, assumed delay 0 | 5 | 3 | 15 |
| C delay only | Delayed rich training | Delay; B training-selected winner physics fixed per seed | 5 | 1 | 5 |
| C joint delay | Delayed rich training | Mass + damping + delay | 5 | 3 | 15 |
| E mismatch | Rich training with omitted joint torque | Mass + damping | 5 | 3 | 15 |
| Total | | | | | 82 |

Additional noiseless sanity tests and reduced-duration CI runs are verification, labeled separately from the 82 evidence fits. Cache immutable completed fits by dataset/config/source/start hashes; do not count the same fit twice across milestones. Complete all three B starts for seed 101 in Milestone 4 so its physical winner exists before Milestone 6. For each seed, delay-only physics comes from that seed's immutable B winner; store the source fit ID and parameter values. It is eligible only if both physical errors are <=10%. Joint-delay fits are eligible only after that seed's delay-only fit is finite and has absolute delay error <=0.002 s, and timing correctness tests pass. If a prerequisite fails, retain it and mark dependent fits not run; do not pretend all 82 completed. Distinguish inaccurate delay due to imperfect fixed physics from an implementation defect. Oracle-fixed delay fixtures may diagnose correctness but are separate labeled tests, not a replacement for the registered evidence.

Mismatch generator: add torque -0.080 * tanh(qvel/0.050) N*m to the hinge using a truth-only dynamics force path at each step. No contacts. Verify the force sign, callback/force isolation, and finite behavior. The fitting model omits this effect. Do not increase mismatch strength after seeing results merely to force parameter bias.

## Baselines and evaluation

For each experiment, training and held-out trajectories get the same evaluator:

- Nominal: mass 0.80, damping 0.04; delay 0.
- Identified: the returned parameters; delay only when the experiment models it.
- Oracle: true mass/damping and true measurement transform in the fitting model family.

In matched/delay studies, the oracle is the generator-consistent noise reference. It is not a strict finite-sample lower bound: a fitted model may partially fit noise. In the mismatch study the three-way oracle retains true mass/damping but omits the extra torque, so it is not the noise floor. Add a separately labeled generator-aware reference with the omitted effect included. Never conflate those two oracles.

Roll out every trajectory from its known initial state for the full horizon. Do not reset from measured state every timestep, fit extra offsets on held-out data, or optimize using held-out controls. Primary metrics compare transformed predictions against noisy measurements. Secondary latent-clean errors use stored clean targets for diagnosis only and are labeled separately.

Per trajectory and split report:

- qpos RMSE, qvel RMSE, and p95 absolute error for each sensor.
- Noise-normalized RMSE = sqrt(mean((e_q/sigma_q)² + (e_v/sigma_v)²) / 2).
- Relative improvement = 1 - RMSE_identified/RMSE_nominal, with null/reason if denominator is zero.
- Split aggregate = sqrt(mean(per-trajectory RMSE²)), giving trajectories equal weight; also retain every trajectory row.
- Training objective and held-out prediction side by side; no held-out start selection.

Parameter rows include estimate/truth/nominal/bounds, absolute/relative errors, seed/start, bound hits, selection status, and diagnostics. A bound hit means distance <= 0.001 times bound width. Consistent bound pinning means >=3/5 selected seed fits hit a bound; report per-start hits too. Delay relative error for zero truth is null; report absolute seconds/timesteps instead.

Across seeds and starts report all rows plus median/min/max, convergence/failure counts, and spread. Five noise seeds support descriptive variation, not a strong population-confidence claim. Save supported official local confidence intervals with assumptions: matched model, near optimum, suitable noise/Jacobian. They are unreliable under mismatch, bounds, correlated delay errors, or poor rank. If a Jacobian is accessible, save scaled-column singular values/rank/condition information (columns scaled by bound width), documenting evaluation point and residual weights. Finite-difference checks can use the official residual; they must not introduce an optimizer. Missing diagnostics are null with reasons.

## Frozen provisional engineering targets

- B matched joint only: both mass and damping errors <=10% in at least 4/5 selected seed results. Report nominal-start recovery separately from the multi-start winner.
- B held-out prediction only: median split qpos and qvel RMSE improvements each >=20% over nominal, and neither held-out trajectory has negative median improvement. Report ablation prediction metrics without requiring those ablations to pass B's target.
- C joint delay: absolute error <=0.002 s in at least 4/5 selected seed results; also report C delay-only performance and whether absolute phase lag decreases relative to zero-delay assumptions.
- B mass/damping and C joint mass/damping/delay: no required parameter is consistently pinned at a bound. Bound hits in weak, delay-excluded, and mismatch fits are diagnostics rather than reasons to hide those results.
- CPU runtime targets: reduced smoke experiment <=60 s; full evidence command <=15 minutes on the recorded local machine, excluding initial dependency installation. These are targets, not runtime predictions.

Missing, invalid, or ineligible seed results count against the fixed five-seed recovery denominator. Aggregate medians are descriptive over available finite rows with counts attached; a target cannot be declared fully verified if a required seed/split is missing. Bound-pinning verification requires complete selected-seed rows. Report scientific failure, execution incompleteness, and diagnostic unavailability separately.

Thresholds are engineering choices to freeze before evidence, not guarantees or statistical significance claims. Failure must remain visible. No positive result is required for weak excitation or mismatch; their scientific questions can yield negative or inconclusive findings.

## Artifacts, provenance, and tests

Required authored files: README.md; these four planning documents; pinned pyproject.toml and uv.lock; model/pendulum.xml; versioned configs/*.json; src/mujoco_sysid_lab/; tests/; scripts/run_experiment.py; scripts/generate_data.py; .github/workflows/ci.yml; .gitignore.

Required generated public artifacts: docs/results/summary.json; docs/results/parameter_table.csv; docs/images/train_trajectory_fit.png; heldout_trajectory_fit.png; residuals.png; parameter_recovery.png; ablation_comparison.png; docs/report.html from official report generation, or a clearly linked official report directory. Public summary includes schema version, registered criteria/results, all seed/start records including failures/not-run, provenance/config references and hashes, metric definitions, diagnostic availability, oracle semantics, and runtime. Curated plots show registered seed 101 plus seed-spread plots, never a silently chosen best seed.

Full datasets, logs, original official reports, optimizer outputs, environment lock/snapshot, and source snapshots stay in ignored artifacts/<run_id>/. Provide an explicit archive command and archive SHA256 so retained public summaries remain auditable even if the working directory is cleaned. Record Git commit/dirty state at run start, distinguish the source commit from any later results-publication commit, and hash the exact config/MJCF/dataset/code/lockfile. Public artifacts must not contain machine serial numbers or private interview material.

Tests required before completion:

1. Same seed/config yields identical numeric arrays and canonical metadata; nondeterministic runtime/run-ID fields excluded from byte-equality assertions.
2. Config validation: finite values, ordered/physical bounds, start within bounds, positive mass/inertia/dt/noise scale, delay support, valid durations/control limits; descriptive errors for invalid ranges.
3. Disjoint trajectory IDs and estimator input isolation; an evaluation/splitting contract that prevents held-out optimization.
4. Correct array dimensions, uniform timestamps, control/sensor endpoint alignment, safe delay sign/window, no extrapolation.
5. Fixed-geometry mass/inertia scaling and stamped fixed physical parameters.
6. One shared metric evaluator for nominal/identified/oracle, checked with known-error fixtures and zero-error cases.
7. Strict report schema, required provenance fields, selected-start logic, retained failures, finite JSON or null/reason for unavailable diagnostics.
8. Reduced-duration official SysID smoke pipeline, official save/report output, and noiseless single-parameter sanity recovery.
9. No NaN/Inf in generated trajectories or summaries; truth-only mismatch force absent from every fitting-model baseline.

CI runs config/unit contracts and a small CPU SysID experiment. Full matrix remains a documented local command unless profiling shows CI can handle it. Local verification and remote CI completion must be reported separately.
