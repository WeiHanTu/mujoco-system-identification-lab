# Milestone 5 — Untouched trajectory prediction

Milestone 5 passed without optimization. The immutable `B_seed101_S2` parameters from [Milestone 4](04_joint_recovery.md) were applied to the previously unused chirp and pulse controls. Held-out position RMSE fell from **0.653406460 to 0.001993802 rad** and velocity RMSE from **2.946468375 to 0.019926075 rad/s**: **99.6949%** and **99.3237%** reductions for this synthetic matched-model seed.

The raw record is `artifacts/milestone5/20261013_seed101/run.json`, SHA256 `f47ed294f2ba6ab2fecf20ce611978c73a31eda30a7f33711c73fb4ae8f94311`. The original training-only selection hash remained `148bea6e1d3407950daabe7daa0bca2f276d2e3c7090b890b8505200e21ba6db`. No model, initial condition, offset, or hyperparameter was refitted on held-out data. [Final summary](../results/summary.json) retains the later five-seed results separately.

## Common metrics

Every prediction is a full eight-second CPU rollout from the known zero position/velocity reset state, using the complete stored controls. Nominal, identified, and oracle models share the same 0.002 s timestep, sensor definitions, [0.060, 7.940] s evaluation window, and metric implementation. Each trajectory contributes 3,941 sensor samples; split RMSE uses equal trajectory weights. Angles are unwrapped and no measured-state reset is used.

| Split | Model | qpos RMSE (rad) | qvel RMSE (rad/s) | Noise-normalized RMSE |
|---|---|---:|---:|---:|
| Training | Nominal | 0.381333326 | 1.670209087 | 147.186569409 |
| Training | Identified | 0.001999670 | 0.020062686 | 1.001486117 |
| Training | Oracle | 0.002000049 | 0.020062063 | 1.001565022 |
| Held-out | Nominal | 0.653406460 | 2.946468375 | 253.415853991 |
| Held-out | Identified | 0.001993802 | 0.019926075 | 0.996602353 |
| Held-out | Oracle | 0.001993022 | 0.019926690 | 0.996422821 |

Training improvement is 99.4756% for position and 98.7988% for velocity. The headline is held-out prediction, not the much lower fitting loss. The identified and oracle predictions are near the registered observation-noise scales of 0.002 rad and 0.020 rad/s.

| Held-out trajectory | Identified qpos RMSE | Identified qvel RMSE | qpos p95 absolute error | qvel p95 absolute error |
|---|---:|---:|---:|---:|
| Chirp | 0.001973981 rad | 0.019715280 rad/s | 0.003896426 rad | 0.038988656 rad/s |
| Pulse | 0.002013427 rad | 0.020134663 rad/s | 0.003944488 rad | 0.039182533 rad/s |

The nominal chirp RMSEs were 0.501000488 rad and 2.423214047 rad/s; pulse RMSEs were 0.776452519 rad and 3.389894608 rad/s. Both trajectories improved separately. Percentiles are NumPy linear per-trajectory p95 values; a mean of percentiles is not reported as a pooled percentile.

The oracle has the true physical parameters but still has measurement-noise residuals. Small finite-sample differences between identified and oracle scores cannot be interpreted as discovering better physics than the generator. This matched simulator family and known actuator gain make the task easier than real trace identification.

## Evidence and reproduction

Evaluation took **0.191424 s**, with **zero optimizer calls**. Its base commit was `a22bab359f1d3450e6df2111a1b9d607af551e40`, and its source tree was **dirty**, as recorded. The complete source snapshot is retained at `artifacts/milestone5/20261013_seed101/source_snapshot`; manifest SHA256 `d7cb71f8dc7dce8ec55909bf2636530f3b9915a0204c97b6b55bcfd5c6b6e642`, tracked patch SHA256 `e15c9250aca0951b16a3e7def64032c075a0e3ce3797f3f67c9cff486c1ba73a`. This evaluation is not described as a clean-commit run. The numerical fit itself is the earlier clean M4 result.

Environment: CPython 3.12.5; MuJoCo 3.15.0; macOS 26.6.2 / Darwin 25.6.0 ARM64; Apple M4 Pro, 12 CPU cores, 24 GB; one rollout worker and one numerical-library thread. Evidence/truth hashes remain `f3e6deacceec9e6365b39e2c9067b9e3a8b71bab6a1c21f6f8e6ada4af824202` and `07c0aac32fe4147921572fdf3e8b134e49c991daa3468de27badb55ce9ef0f2f`. Each prediction NPZ includes controls, timestamps, measured/clean observations, and all three predictions, with file/array hashes in the raw record.

The recorded command was `.venv/bin/python scripts/evaluate_milestone5.py --output artifacts/milestone5/20261013_seed101`. For a fresh evaluation, preserve the historical directory and run:

```sh
.venv/bin/python scripts/evaluate_milestone5.py
```

The frozen success target requires five-seed median improvement of at least 20% for both held-out sensor RMSEs, with neither held-out trajectory having negative median improvement. This seed-101 checkpoint alone does not establish that later target. It establishes held-out synthetic prediction for the already selected physical model. It provides no real-robot, HIL, or deployment evidence.
