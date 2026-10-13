# Milestone 8 — Better prediction with biased physics

Milestone 8 completed all **15 registered mismatch fits** across five seeds and three starts. Every fit converged, and every selected damping estimate was biased upward by **37.45–37.50%**, while held-out position/velocity predictions still improved by about **96.6% / 96.1%** over nominal. This supports compensation for model misspecification: a better prediction can coexist with incorrect physical parameters.

The raw source is `artifacts/full/20261013_evidence_v1/run.json`, SHA256 `c078ecda4e479549eb8d1d87d37a961d6282c418eca1d5d76266f5f5a13fb67b`. [summary.json](../results/summary.json) and [parameter_table.csv](../results/parameter_table.csv) retain every start, parameter error, training/held-out metric, and diagnostic.

## Generator and fitting model

Only the target generator applies the registered extra torque `-0.080*tanh(qvel/0.050)` N*m at the start of each control interval, held over that interval. This is a smooth Coulomb-like dissipative effect, not a contact simulation or MuJoCo dry-friction parameter. The force is local `MjData.qfrc_applied`; no global callback or fitting-model mutation is introduced. The terminal recorded torque is unused.

True mass and viscous damping remain 1.20 kg and 0.15 N*m*s/rad. Estimation still fits only mass and viscous damping, holding actuator gain, geometry, COM, gravity, initial state, and delay known. Noise innovations are paired with the matched study. The optimizer sees sanitized rich training records; the omitted force and clean truth are absent from its inputs. Chirp/pulse evaluation follows sealed training-only selection.

| Selected-seed statistic | Median | Minimum | Maximum |
|---|---:|---:|---:|
| Mass estimate (kg) | 1.209930256 | 1.209566222 | 1.210063663 |
| Mass relative error | 0.827521% | 0.797185% | 0.838639% |
| Damping estimate (N*m*s/rad) | 0.206221874 | 0.206178185 | 0.206250534 |
| Damping relative error | 37.4812% | 37.4521% | 37.5004% |
| Training NRMSE | 5.519636 | 5.513313 | 5.525737 |
| Held-out NRMSE | 8.973470 | 8.956753 | 8.978632 |
| Held-out qpos improvement | 96.6214% | 96.6165% | 96.6251% |
| Held-out qvel improvement | 96.1042% | 96.0902% | 96.1127% |

The residual is much smaller than nominal but remains clearly above the observation-noise reference. Do not describe this as a noise-level fit with wrong physics. The selected seed-101 training objective is 239,683.0111, compared with 199,166,554.154 for the fully biased nominal S0 model. Held-out prediction is worse than training fit, consistent with compensation that transfers imperfectly across excitation.

## Two reference meanings

| Seed-101 held-out model/reference | qpos RMSE (rad) | qvel RMSE (rad/s) | NRMSE |
|---|---:|---:|---:|
| Identified fitting model | 0.022538718 | 0.117018337 | 8.978632 |
| Fitting-family oracle: true mass/damping, omitted torque absent | 0.052168938 | 0.255559042 | 20.538690 |
| Generator-aware clean-observation reference: omitted torque included | 0.001993022 | 0.019926690 | 0.996423 |

The identified fitting model outperforms the fitting-family oracle because biased effective damping helps approximate the omitted torque. The three-way oracle therefore is not the noise floor in this scenario. The separately labeled generator-aware reference compares measured signals with the actual noise-free target observations, including the omitted effect and measurement transform; it isolates observation noise and is not an estimated model. Even that reference is not a strict finite-sample lower bound for a fit that can absorb noise.

## Interpretation and diagnostics

All three starts at each seed converged to closely agreeing effective parameters, without bound hits. Selected scaled-Jacobian condition numbers are about 1.233 and rank is 2. Good local sensitivity and convergence within the chosen model do not establish that the model contains the correct physics. The stable bias, known omitted dissipative torque, residual above the generator-aware reference, and agreement across starts support model misspecification as the explanation here.

This does **not** prove structural non-identifiability, and convergence alone does not rule out every optimizer issue. The shared residual/objective checks, retained native results, distinct model-force paths, and multiple starts make an unexplained implementation or gross local-fit failure less plausible in this controlled experiment. Official local confidence intervals remain conditional on matched dynamics/local linear assumptions; they are unreliable as physical uncertainty intervals under this omitted-force mismatch, even if numerical interval computation succeeds.

No mismatch amplitude or acceptance threshold was tuned to strengthen the narrative. The result illustrates a limitation of physical interpretation under a deliberately known misspecification, not real-world validity or arbitrary-robot identifiability.

## Provenance and reproduction

All E fits were newly executed in the clean full-study invocation at commit `ec50b46b65344207915f33fc1db7ed47ddfc0ae7`. Total full invocation time was **57.173810 s** with 70 new/12 reused fits; no separate E-only duration is asserted. The raw record contains each E fit's own runtime, evaluation counts, objective trace, native arrays, confidence/sensitivity diagnostics, official report/model hashes, exact source/environment, and command.

Environment: CPython 3.12.5; MuJoCo 3.15.0; macOS 26.6.2 / Darwin 25.6.0 ARM64; Apple M4 Pro, 12 CPU cores, 24 GB; one rollout worker and one numerical-library thread. Evidence/truth config hashes remain `f3e6deacceec9e6365b39e2c9067b9e3a8b71bab6a1c21f6f8e6ada4af824202` and `07c0aac32fe4147921572fdf3e8b134e49c991daa3468de27badb55ce9ef0f2f`.

The retained command was `.venv/bin/python scripts/run_experiment.py --config configs/evidence_v1.json --milestone 9 --output artifacts/full/20261013_evidence_v1`. For a fresh run, omit the dated output destination:

```sh
.venv/bin/python scripts/run_experiment.py --config configs/evidence_v1.json --milestone 9
```

The project is a reproducible synthetic dynamics system-identification study. Real robot data, HIL, physical actuator calibration, contact-rich manipulation, real sim-to-real transfer, and production deployment are not demonstrated.
