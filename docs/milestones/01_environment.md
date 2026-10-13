# Milestone 1 verification

Verified locally on 2026-10-12, America/Los_Angeles. This checkpoint covers environment, API contracts, and deterministic synthetic simulation. It does not cover identification, held-out prediction, or real hardware.

## Evidence

- Source commit: `4ad8657763b7568e1f16d92d3a3808cb739488c9`; clean at run start.
- Python 3.12.5; mujoco[sysid] 3.15.0; all 36 installed packages and lock/source fingerprints are recorded in [milestone1.json](../results/milestone1.json).
- CPU environment: Apple M4 Pro, 12 logical cores, 24 GB; macOS 26.6.2. One numerical-library thread and one official rollout worker.
- Seed 101 is recorded. The one-second sanity trajectory has no noise and uses no randomness.
- 501 samples, 500 control intervals, 0.002 s timestep. Repeated timestamps/controls/qpos/qvel/sensors are numerically identical; all values finite; direct sensor/state error is zero.
- Warm-cache full verification: 0.6805 s. Two one-second rollouts: 0.001816 and 0.001733 s. Full-study runtime remains unknown.
- Local tests: 86 passed in 1.60 s, with 16 upstream NumPy/Pyparsing deprecation warnings visible. Warnings were not suppressed; remote CI has not run.

Exact retained-run command:

```sh
.venv/bin/python scripts/verify_milestone1.py --config configs/milestone1.json --output artifacts/milestone1/20261012_seed101_verified
```

The runner records exact config, verification pulses, package versions, hardware/software/thread settings, Git state, source hashes, API source signatures/fingerprints, raw NPZ/plot hashes, and completed setup-input hashes. Each mutable console log is retained separately, rather than hashed as an input during execution.

## Installed API findings

Required symbols and actual signatures are available. The official MjSpec/ParameterDict/TimeSeries/ModelSequences/build_residual_fn pipeline executes without calling optimize. A narrow supported modify_residual hook produces an unchanged-model residual of zero and matches direct measurement-noise whitening within 6.94e-15.

The released package's differences from simple documentation examples are explicit:

1. Import may hide missing optional dependencies; required exports are checked individually.
2. Frozen physical modifiers are skipped. Fixed physics must be stamped into the spec.
3. Body mass helpers do not scale rotational inertia by default; explicit scaling was verified.
4. The normalizer docstring says standard deviation; code uses per-sensor L2 norm divided by sqrt(2). It is disabled for pure 1/sigma weighting.
5. The residual example's signal_transform argument is unsupported; modify_residual is the supported hook.
6. Jointpos/jointvel rollout values precede integration but receive post-step timestamp labels in SysID. Fresh observations are reconstructed from FULLPHYSICS states, with an explicit initial observation. The measured one-step offset is 0.002 s.
7. Positive-delay automatic windowing admits unsupported leading samples. A predeclared fixed safe window is enforced; sign/support tests cover 0, 0.024, and 0.060 s.
8. An eight-second timing-only test checks clock accumulation roundoff and retains both delay-window boundaries. This is not a generated study dataset or identification result.
9. The official scipy_parallel_fd backend maps max_iters to max_nfev and supplies bounded MuJoCo finite differences. Backend dispatch was intercepted in a test; no estimator ran.
10. save_results and default_report are separate. Saving optimizer results, computing intervals, and generating official HTML are deferred to the identification milestone.

The alignment hook is restricted to the authored one-hinge, two-sensor layout. It rejects other sensor/joint semantics and unsafe support. It does not replace the official optimizer or silently estimate a numerical timestamp shift as physical delay.

## Retained audit failure

The first run's numerical checks passed, but its setup manifest incorrectly hashed a console log still being written. Its original run.json, trajectory, plot, and separate failed audit.json remain under artifacts/milestone1/20261012_seed101/. The defect was fixed, a regression check added, and the verified run executed from a new clean source commit. Thresholds and mechanics were unchanged.

The local archive includes both runs and completed setup/test/console logs. Its real SHA256 and tar command are in the public JSON record. Data/plot hashes, five setup-input hashes, and 22 source-file hashes against the recorded Git commit passed the final audit.

## Checkpoint

Milestone 1 passed. Stop before Milestone 2. Next work is the registered ground-truth/nominal synthetic datasets, noise metadata, trajectory split, and plots; no optimization starts there.
