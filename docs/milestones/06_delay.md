# Milestone 6 — Shared sensor delay

Milestone 6 passed on seed 101. A hidden **0.024 s** shared position/velocity delay was recovered first with the immutable B winner's physics fixed, then with mass, damping, and delay free. The delay-only estimate was **0.023985576827904972 s** (14.423 microseconds absolute error); the selected joint estimate was **0.023984869571342996 s** (15.130 microseconds error). Both are inside the unchanged one-timestep target of **0.002 s**.

The raw record is `artifacts/milestone6/20261013_seed101/run.json`, SHA256 `8fec39214147d7abde9ecbd2669527ae2cf7b5d704f42c02d089354a796a0ce4`. It retains seven newly executed registered delay-study fits and three reused seed-101 B starts, with their original provenance. Earlier A1/A2 are separate completed evidence, not rerun here. [Final summary](../results/summary.json) includes all five seeds.

## Timing contract and prerequisites

Measured signals follow `y(t)=clean(t-delay)+noise(t)`. The generator retains explicit [-0.060, 0] s zero-control equilibrium history; noise is added after the clean observation transform. The same deterministic innovations are paired with matched, delayed, and mismatch scenarios. The fitting objective uses the fixed inclusive [0.060, 7.940] s observation window for every candidate, including zero delay. It never crops samples according to a candidate or extrapolates beyond support.

Delay-only physics is fixed to `B_seed101_S2` from Milestone 4, mass 1.1998852426162079 kg and damping 0.1500286351169055 N*m*s/rad. Both physical errors passed the registered 10% eligibility check. Delay-only starts at 0.002 s. Only after its finite result met the 0.002 s timing target were all three joint-delay starts run. Joint starts are paired with S0/S1/S2 delays 0.002/0.050/0.010 s; their bounds remain [0, 0.060] s. Start selection uses training objective only.

Reduced timing verification exposed a MuJoCo 3.15.0 windowing edge case: `SignalTransform.delay` can crop a valid canonical endpoint when, for example, the grid ends the safe interval at `1.1400000000000001` but raw subtraction yields `1.14`. After an explicit support check at both registered delay bounds, the alignment hook now uses official `sysid.apply_resample_and_delay` on the already fixed measured grid; official `SignalTransform` retains the unchanged noise-weighted residual semantics. This corrects API/timestamp plumbing rather than adding an optimizer or changing a scientific objective. Sign, endpoint, zero/max delay, and fixed-sample tests passed before the retained delay evidence. The 0.002 s recovery criterion and frozen configs were unchanged.

## Prediction and compensation

| Seed-101 delayed-data model | Mass (kg) | Damping (N*m*s/rad) | Delay (s) | Training NRMSE | Held-out NRMSE |
|---|---:|---:|---:|---:|---:|
| Physical fit assuming zero delay | 1.177968157 | 0.159024829 | 0 | 11.740180 | 14.265219 |
| Delay-only; B physics fixed | 1.199885243 | 0.150028635 | 0.023985577 | 1.001468 | 0.996422 |
| Joint physical + delay | 1.199874399 | 0.150032002 | 0.023984870 | 1.001468 | 0.996448 |
| Oracle | 1.20 | 0.15 | 0.024 | Near noise reference | 0.996423 |

Without delay, physical fitting partially compensates: mass is 1.836% low and damping 6.017% high. That model still improves substantially over the biased nominal model, yet retains much larger residuals than the delay-aware fits. This is measured parameter compensation for an omitted timing effect, not evidence that the recovered zero-delay physics is correct.

The delay-only held-out RMSEs are 0.001993328 rad and 0.019923582 rad/s. Joint held-out RMSEs are 0.001993437 rad and 0.019923563 rad/s. These compare transformed predictions with noisy observations using the same evaluator as Milestone 5; secondary instantaneous-state diagnostics remain separate.

The registered phase diagnostic correlates demeaned `p(t)` with `y(t+lag)` on fixed [0.120, 7.880] s samples, searching -0.060 to +0.060 s in 0.002 s increments. Positive lag means measurements lag prediction. Ties within 1e-12 prefer the smallest absolute lag, then negative lag; signals below noise sigma or peaks below 0.20 return null with a reason. With exactly the same fixed B physics, median absolute lag is **0.024 s before correction and 0 s afterward**, on both training and held-out splits. Zero on this grid means no resolved lag at the diagnostic's 2 ms resolution; it is not an uncertainty interval or exact proof of zero timing error.

## Provenance and reproduction

The checkpoint used clean commit `ec50b46b65344207915f33fc1db7ed47ddfc0ae7` and measured **6.068619 s**, including seven new fits, cached B verification, official reports, and evaluation. Environment: CPython 3.12.5; MuJoCo 3.15.0; macOS 26.6.2 / Darwin 25.6.0 ARM64; Apple M4 Pro, 12 CPU cores, 24 GB; one rollout worker and one numerical-library thread. Complete native arrays, call ledgers, report/model hashes, and cache provenance are retained in the raw record.

Evidence/truth config hashes remain `f3e6deacceec9e6365b39e2c9067b9e3a8b71bab6a1c21f6f8e6ada4af824202` and `07c0aac32fe4147921572fdf3e8b134e49c991daa3468de27badb55ce9ef0f2f`. See [registration](../results/registration_v1.json).

The historical command included `--output artifacts/milestone6/20261013_seed101`. A fresh invocation preserves that directory:

```sh
.venv/bin/python scripts/run_experiment.py --config configs/evidence_v1.json --milestone 6
```

These are synthetic timing and prediction checks. They demonstrate no physical sensor calibration or hardware timing validation. Five-seed timing results are in [Milestone 7](07_robustness.md).
