# Milestone 4 — Joint mass and damping recovery

Milestone 4 passed. All three registered seed-101 starts converged with MuJoCo 3.15.0's official SysID optimizer. The training-only selection chose `B_seed101_S2`: mass **1.1998852426162079 kg** and damping **0.1500286351169055 N*m*s/rad**. Their relative errors against the evaluation-only truth are **0.00956312%** and **0.01909008%**; neither is at a registered bound. This was one matched-model seed, not the five-seed robustness conclusion.

The immutable raw record is `artifacts/milestone4/20261013_seed101/run.json`, SHA256 `5fdd819cff493dec74f6e02d2ef7fb263980154fef0ee41c448962e3b0777d87`. Final project-wide results are in [summary.json](../results/summary.json) and [parameter_table.csv](../results/parameter_table.csv).

## Training selection and results

Only the two complete rich training trajectories enter estimation or selection. Both run for eight seconds at a 0.002 s timestep. The common [0.060, 7.940] s sensor window supplies 15,764 whitened residual entries. The linear objective is `0.5*sum(((measured-predicted)/sigma)**2)`, with sigma 0.002 rad for position and 0.020 rad/s for velocity. Amplitude normalization is disabled.

| Start | Initial mass / damping | Initial objective | Final objective | Native nfev / njev | Callback events | Fit runtime (s) |
|---|---|---:|---:|---:|---:|---:|
| S0 | 0.80 / 0.04 | 170,754,751.1426003 | 7,905.444560284710 | 9 / 9 | 8 | 0.367528 |
| S1 | 1.70 / 0.30 | 13,288,216.2101461 | 7,905.444560284701 | 5 / 5 | 4 | 0.218766 |
| S2 | 0.50 / 0.40 | 50,106,138.3777918 | 7,905.444560284699 | 7 / 7 | 6 | 0.295164 |

All three finite outcomes are retained. Selection uses the exact minimum final training objective, including a finite nonconverged result if one existed. Exact ties prefer S0, then S1, then S2. The observed cost differences are around 1e-11: S2 is the recorded winner, but there is no meaningful performance advantage over the other starts. Their recovered mass values differ by less than 4e-11 kg and damping by less than 1.1e-11 N*m*s/rad.

The selected record was sealed before loading physical truth for parameter-error evaluation. The selection SHA256 is `148bea6e1d3407950daabe7daa0bca2f276d2e3c7090b890b8505200e21ba6db` and remained unchanged afterward. No held-out score was calculated in this milestone. The five cumulative registered fits at this checkpoint are A1, A2, and these three B starts; noiseless verification is counted separately.

| Training model | qpos RMSE (rad) | qvel RMSE (rad/s) | Noise-normalized RMSE |
|---|---:|---:|---:|
| Nominal | 0.381333326 | 1.670209087 | 147.186569409 |
| Selected identified | 0.001999670 | 0.020062686 | 1.001486117 |
| Oracle | 0.002000049 | 0.020062063 | 1.001565022 |

These shared-evaluator metrics reconstruct the official final objective. An identified finite-sample fit can score slightly below the oracle by fitting some observation noise; the oracle is a controlled reference, not a strict finite-sample bound.

## Diagnostics and gate

The selected bound-width-scaled residual Jacobian has shape [15,764, 2], rank 2, and condition number **1.34103663**. This supports local separation of the two free parameters under these excitations and fixed mechanics; it does not prove general identifiability. Known motor gain, geometry, COM, gravity, initialization, armature, and absent dry friction remain fixed. Mass changes scale every COM inertia component.

Official 95% local Student-t intervals are [1.1996901983, 1.2000802870] kg and [0.1499820471, 0.1500752231] N*m*s/rad. They assume matched dynamics, suitable local linear sensitivity, and independent observation errors. They do not measure five-seed robustness or real-world uncertainty. Native `nit` is unavailable; callback events and native function/Jacobian evaluations are reported separately. Finite-difference rollout work and later report evaluations have separate call ledgers.

All three official reports saved successfully. Compiled MJB preserves full-precision mechanics, while official XML retains six-significant-digit serialization with recorded rounding error; parameter YAML and native optimizer results retain the fitted values. No scientific threshold was relaxed. The independent audit passed **142 checks**, including native arrays, objective/metrics, sensitivity/intervals, selection, saved models, source hashes, and frozen configuration. Its ignored `audit.json` SHA256 is `550a12f47e2ae0e78c7fca46d6621e350b0156bc257f1a09941f351ec7f7e3b0`.

Both selected physical errors satisfy the pre-registered 10% eligibility test for the later delay-only stage. Held-out improvement is evaluated in [Milestone 5](05_heldout_prediction.md), without refitting.

## Provenance and reproduction

The run used clean source commit `a22bab359f1d3450e6df2111a1b9d607af551e40` and measured **2.049861 s** for fitting, reports, and training evaluation. Environment: CPython 3.12.5; MuJoCo 3.15.0; macOS 26.6.2 / Darwin 25.6.0 ARM64; Apple M4 Pro, 12 CPU cores, 24 GB; one rollout worker and one numerical-library thread. Exact package versions, command, file/array hashes, and source hashes are in the raw record.

The unchanged evidence config SHA256 is `f3e6deacceec9e6365b39e2c9067b9e3a8b71bab6a1c21f6f8e6ada4af824202`; the truth config SHA256 is `07c0aac32fe4147921572fdf3e8b134e49c991daa3468de27badb55ce9ef0f2f`. [Registration](../results/registration_v1.json) contains the exact snapshots, bounds, starts, options, and criteria.

The retained command included `--output artifacts/milestone4/20261013_seed101`. For a fresh run, leave the output unset so existing evidence is preserved:

```sh
.venv/bin/python scripts/run_experiment.py --config configs/evidence_v1.json --milestone 4
```

This is synthetic parameter recovery only. It demonstrates no hardware, HIL, actuator calibration, or real sim-to-real transfer.
