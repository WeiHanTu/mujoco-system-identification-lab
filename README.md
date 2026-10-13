# mujoco-system-identification-lab

A reproducible synthetic dynamics system-identification study that exercises the estimation, validation, and diagnostic workflow needed before applying the method to real traces.

Completed: Milestone 3 independent mass and damping recovery with MuJoCo's official SysID toolbox. On noise seed 101, mass error was **0.00836%** and damping error was **0.01689%**. Each fit was oracle-assisted: the other physical parameter was fixed to its true value. This verifies the individual parameter paths; joint recovery and identified-model held-out improvement remain unverified. The local suite passed **326 tests**.

| Oracle-assisted fit | Start | True value | Recovered value | Relative error | Bound hit |
|---|---:|---:|---:|---:|---|
| Mass; true damping fixed | 0.80 kg | 1.20 kg | 1.199899661 kg | 0.00836% | No |
| Damping; true mass fixed | 0.04 N*m*s/rad | 0.15 N*m*s/rad | 0.150025342 N*m*s/rad | 0.01689% | No |

![Independent physical parameter recovery](docs/images/milestone3_parameter_recovery.png)

[Milestone 3 verification](docs/milestones/03_single_parameter_recovery.md) · [Machine-readable evidence](docs/results/milestone3.json) · [Parameter rows](docs/results/milestone3_parameter_table.csv) · Official reports: [mass](docs/reports/milestone3/A1/report.html), [damping](docs/reports/milestone3/A2/report.html)

Two registered evidence fits out of 82 are complete, alongside two separately labeled noiseless sanity checks. All four used the complete multisine and piecewise-constant training trajectories. The chirp and pulse trajectories did not enter fitting or fitted-model scoring at this milestone.

## Reproduce Milestone 3

Run from the repository root:

```sh
uv sync --locked --python 3.12.5
.venv/bin/python scripts/run_experiment.py --config configs/evidence_v1.json --milestone 3
.venv/bin/python -m pytest
```

Leave `--output` unset so the runner creates a fresh UTC-timestamped directory under `artifacts/milestone3/`. Existing evidence directories are rejected. This command runs the two independent noisy fits and two noiseless checks, saves native results, and generates official HTML with videos disabled. It does not run the remaining milestones.

The noisy fits reduced the training objective from 15,888,124 to 7,906.17 for mass and from 25,239,640 to 7,906.11 for damping. The objective is `0.5*sum(noise_normalized_residual**2)` over 15,764 residual entries. Training noise-normalized RMSE was 1.00153 for both fitted unit models, close to the matched oracle's 1.00157. These are training diagnostics; the study's primary held-out target has not been assessed.

Both noisy fits converged: mass terminated on `ftol` and `xtol`, damping on `ftol`. Each recorded seven native function evaluations, seven Jacobian evaluations, and six callback events. SciPy does not expose an iteration count here; callback events are reported separately. The noiseless checks recovered mass and damping to relative errors 4.07e-15 and 7.40e-16, below the unchanged 0.1% numerical sanity threshold.

The original optimization batch used clean source `5550a43b1d36f0be1820e1f9c3d39b2eabdc1bda` and took 2.179117 s. Its noisy fits converged, but report verification failed because official XML rounds physical fields to six significant digits. That failure remains retained. A separate clean-source report repair at `363241a960ed702d481e4ad58e103ac943341245` took 1.259173 s, reused all four native optimization results, and reproduced their summaries exactly without rerunning optimization. These runtimes describe separate invocations.

XML verification now compares to the verified six-digit serialization target at unchanged `rtol=atol=1e-12`. Full-precision estimates remain in parameter YAML and optimizer pickle; an official MuJoCo MJB model preserves the compiled physical fields exactly. Parameter estimates, configs, and scientific thresholds did not change. Raw initial and repaired runs, all four reports, logs, arrays, and source provenance are archived locally in ignored `artifacts/milestone3/20261012_evidence_verified.tar.gz`; its actual hash and archive command are in [milestone3.json](docs/results/milestone3.json).

The environment remains Python 3.12.5, MuJoCo 3.15.0, macOS 26.6.2, Apple M4 Pro (12 cores, 24 GB), one rollout worker, and one numerical-library thread. The final local suite passed 326 tests in 4.47 s with 148 upstream deprecation warnings visible. Remote CI and full-study runtime remain unverified.

Milestone 3 passed. Stop before Milestone 4 joint mass/damping recovery.

## Deterministic synthetic data

![Training measurements and nominal/oracle baselines](docs/images/milestone2_training_baselines.png)

[Milestone 2 verification](docs/milestones/02_synthetic_data.md) · [Machine-readable evidence](docs/results/milestone2.json) · [Baseline rows](docs/results/milestone2_baseline_metrics.csv) · [Frozen registration](docs/results/registration_v1.json)

The authored model is a single actuated hinge with a uniform rigid link. Ground-truth mass is 1.20 kg and damping is 0.15 N*m*s/rad; the nominal model starts at 0.80 kg and 0.04 N*m*s/rad. Geometry, COM, gravity, motor gain, and initial state are known. Mass and COM inertia scale together. The timestep is 0.002 s and all rollouts run on CPU.

## Reproduce Milestone 2

Run from the repository root with the locked environment:

```sh
uv sync --locked --python 3.12.5
.venv/bin/python scripts/generate_data.py --config configs/evidence_v1.json
.venv/bin/python -m pytest
```

Leave `--output` unset. Each invocation creates a fresh UTC-timestamped directory under `artifacts/milestone2/`; existing output directories are rejected. The generator writes per-trajectory NPZ, a JSON manifest, nominal/oracle metric rows, four plots, source/environment provenance, and the frozen registration. It generates matched seed-101 data by default and performs no optimization.

The training split contains multisine and seeded piecewise-constant controls. The held-out split contains a chirp and pulses with different amplitudes. Splitting uses complete trajectory IDs. The training reader loads only controls, noisy observations, timestamps, and the known initial state; it excludes truth, clean targets, nominal predictions, and held-out arrays. Hidden truth is kept in a separate generator/evaluation config.

Observation noise is independent Gaussian with sigma 0.002 rad for position and 0.020 rad/s for velocity. The matched dataset has zero delay. A saved equilibrium prehistory from -0.060 to 0 s and a 0.10 s zero-control prefix support later delay experiments. All baselines use the same fixed evaluation window [0.060, 7.940] s, containing 3,941 samples.

| Split | Model | qpos RMSE (rad) | qvel RMSE (rad/s) | Noise-normalized RMSE |
|---|---|---:|---:|---:|
| Training | Nominal | 0.381333 | 1.670209 | 147.186569 |
| Training | Oracle | 0.002000 | 0.020062 | 1.001565 |
| Held-out | Nominal | 0.653406 | 2.946468 | 253.415854 |
| Held-out | Oracle | 0.001993 | 0.019927 | 0.996423 |

Each split aggregates per-trajectory RMSE with equal trajectory weight. The oracle uses the generator's physical parameters, so its residual contains observation noise. It is a controlled reference, not a strict finite-sample lower bound. These are baseline diagnostics; they do not demonstrate identified-model improvement or satisfy the later recovery criteria.

![Held-out measurements and nominal/oracle baselines](docs/images/milestone2_heldout_baselines.png)

The verified run used clean source commit `fa40206800643ff9e0b44bafed02a036ab47a3a3`, Python 3.12.5, MuJoCo 3.15.0, macOS 26.6.2, and an Apple M4 Pro (12 cores, 24 GB). Generation, repeated-array/metadata checks, NPZ round-trip, metrics, and plots completed in 1.290060 s. The suite passed 270 tests in 4.41 s with 17 upstream deprecation warnings visible. Remote CI has not run; full-study runtime remains unknown.

The evidence config and separate truth config were frozen before the first identification run. Their exact SHA256 values, all 82 registered fits, five seeds, three starts, delay prerequisites, noise keys, metrics, and unchanged acceptance criteria are in [registration_v1.json](docs/results/registration_v1.json). The seed-101 matched dataset supplied the two completed independent evidence fits; the other 80 registered fits remain later work.

Raw evidence is archived locally under ignored `artifacts/milestone2/20261012_evidence_verified.tar.gz`. Its actual hash and creation command are recorded in [milestone2.json](docs/results/milestone2.json). The original run remains retained: its numerical data were valid, but plot paths in its report and the explicit mapping from the registered `noise_normalized_rmse` name to the serialized `normalized_rmse` field needed correction. Regression checks and a new clean-source run corrected those metadata defects without changing configs, numeric data, or thresholds. The first archive is also preserved; the verified archive disables macOS metadata files added by default tar behavior.

Milestone 2 passed; its retained baseline metrics remain separate from the later oracle-assisted single-parameter results.

## Reproduce Milestone 1

With the existing locked environment, run from the repository root:

```sh
.venv/bin/python scripts/verify_milestone1.py --config configs/milestone1.json
```

Leave `--output` unset for repeat runs. Each invocation creates a fresh UTC-timestamped directory under `artifacts/milestone1/`. An explicit output directory must be new; the runner refuses to overwrite retained evidence. The dated explicit-output command in the milestone record documents a historical run and is not a rerun command.

Python 3.12.5 is the recorded interpreter. Direct dependencies and every resolved runtime dependency are pinned in pyproject.toml and uv.lock. The project has no dependency on MuJoCo Git main.

```sh
uv sync --locked --python 3.12.5
uv run --locked python scripts/verify_milestone1.py --config configs/milestone1.json
uv run --locked pytest
```

Run from a committed checkout. The verification command refuses to retain results without a real Git commit. It writes a new ignored artifacts/milestone1/<run_id>/ directory containing strict JSON provenance/API checks, raw trajectory NPZ, and a simulation plot. It records dirty state and snapshots dirty authored source. Existing output directories are never overwritten.

The retained Milestone 1 evidence run used clean source commit `4ad8657763b7568e1f16d92d3a3808cb739488c9`, Python 3.12.5, MuJoCo 3.15.0, macOS 26.6.2, and an Apple M4 Pro (12 cores, 24 GB). It completed verification in 0.681 s with a warm font cache; the two one-second simulated rollouts took 0.00182 and 0.00173 s. Its original local suite passed 86 tests. This does not predict full-study runtime. Remote CI has not run.

[Milestone 1 verification](docs/milestones/01_environment.md) · [Milestone 1 evidence](docs/results/milestone1.json)

Raw evidence is archived locally in ignored artifacts/milestone1/20261012_evidence.tar.gz, including the earlier run's retained audit failure. The archive SHA256 and exact creation command are in docs/results/milestone1.json. The first run hashed its still-active console log; a regression test and a new source commit corrected that provenance defect. No physical parameters or acceptance thresholds changed.

The verification trajectory has two small torque pulses and runs twice without noise. The seed is recorded, but this noiseless check uses no randomness. Tests separately cover equilibrium, one-step analytic dynamics, control/sensor timing, inertia scaling, invalid configs/bounds, and provenance. Local determinism does not imply bitwise equality across platforms.

## Installed SysID integration

The implemented single-parameter estimator uses official MjSpec, bounded ParameterDict, TimeSeries, ModelSequences, build_residual_fn, sysid.optimize, save_results, and default_report. The registered solver is the toolbox's official scipy_parallel_fd backend; it retains MuJoCo rollout/Jacobian evaluation while providing SciPy termination diagnostics.

The released package requires explicit handling of several contracts: frozen physical modifiers are skipped; mass helpers do not scale inertia by default; the default residual adds amplitude scaling; optimizer max_iters maps to a function-evaluation budget; and HTML reporting is separate from save_results. Installed source fingerprints and runtime probes document these behaviors.

For this exact jointpos/jointvel sensor pair, MuJoCo rollout sensor output precedes integration, while the SysID conversion uses post-step state timestamps. A narrow supported modify_residual hook reconstructs these two observations from the returned integration states, with explicit initial observation and a fixed safe delay window. This preserves physical timestamps rather than absorbing a numerical offset into the estimated delay. It is deliberately limited to the authored one-DOF sensor layout.

Pure noise weighting disables the upstream amplitude normalization. The official residual sign is measured minus predicted; scalar least-squares loss and RMSE are unchanged by this sign convention. No alternate optimizer is implemented.

Official report terminology needs care. Its observation legend `nominal` denotes measured observations; its parameter-table errors compare against the starting reference rather than hidden truth. Upstream bound highlighting also differs from this project's rule of 0.001 times bound width. The project fit summaries provide authoritative truth errors and bound hits. A visible interpretation section in each official report explains these distinctions.

The exported uncertainty function is `sysid.calculate_intervals`, not `compute_param_intervals`. Its 95% intervals are local Student-t approximations with estimated residual variance and the other physical parameter treated as known. They require a matched model, independent observation noise, and suitable local sensitivity; they do not establish joint identifiability, robustness across seeds, or real-world confidence. Noiseless checks report statistical intervals as unavailable.

## Planned evidence

The registered study contains 82 fits: independent physical checks, joint recovery, weak excitation, omitted/modeled delay, five noise seeds, three initializations, and model mismatch. Training uses multisine and piecewise-constant excitation; chirp and pulse controls remain completely held out. Start selection uses training objective only.

The primary result will be held-out synthetic prediction improvement. Every seed/start and failure will remain visible. The weak step is allowed to identify both parameters; model mismatch is allowed to yield an inconclusive bias result. Neither ablation is tuned to force a preferred story.

See [spec.md](spec.md) for exact configuration, metrics, thresholds, noise keys, solver options, and oracle definitions; [plan.md](plan.md) for all nine milestone gates; [intend.md](intend.md) for claim boundaries.

## Claim boundaries

This project is synthetic workflow validation. The matched generator and fitting model use the same simulator family.

Verified so far: released package imports, inspected/executed API contracts, deterministic CPU simulation and synthetic data generation, mass/inertia consistency, sensor timing/noise-weighting checks, complete-trajectory split isolation, nominal/oracle baseline metrics on both splits, and independent oracle-assisted physical recovery for seed 101. Joint mass/damping recovery, fitted held-out prediction improvement, excitation/delay/initialization sensitivity, and model-mismatch results require later evidence.

Not demonstrated: real robot data, real-robot validation, hardware-in-the-loop, physical actuator calibration, contact-rich manipulation, real sim-to-real transfer, production deployment, or general parameter identifiability for arbitrary robots.

All models and data are authored synthetic assets. Raw runs stay in ignored artifacts/. Bilingual interview notes stay in ignored private/ and must not be published to GitHub.
