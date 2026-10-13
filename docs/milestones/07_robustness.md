# Milestone 7 — Seeds, starts, and excitation

Milestone 7 passed the registered matched-model and timing targets across all five noise seeds, 101–105. All starts are retained and each seed's winner is selected by training objective alone. The complete study later finished **82/82 unique registered fits**, all converged, including the 15 mismatch fits documented in [Milestone 8](08_model_mismatch.md). No missing row, failure, or nonconverged candidate was discarded.

The raw full-run record is `artifacts/full/20261013_evidence_v1/run.json`, SHA256 `c078ecda4e479549eb8d1d87d37a961d6282c418eca1d5d76266f5f5a13fb67b`. [summary.json](../results/summary.json) and [parameter_table.csv](../results/parameter_table.csv) retain every start and the final criteria. Five deterministic noise seeds describe this controlled study; they are not a population reliability estimate.

## Registered matrix and outcomes

| Cell | Seeds | Starts per seed | Retained fits |
|---|---:|---:|---:|
| A1/A2 independent oracle-assisted checks | Seed 101 | One each | 2 |
| B matched mass+damping | 5 | 3 | 15 |
| Weak step mass+damping | 5 | 3 | 15 |
| Delay excluded on delayed data | 5 | 3 | 15 |
| Delay-only; same-seed B winner physics fixed | 5 | 1 | 5 |
| Joint mass+damping+delay | 5 | 3 | 15 |
| E omitted-force mismatch | 5 | 3 | 15 |
| Total | | | 82 |

The two additional noiseless unit checks and reduced CI probes are outside these 82 evidence fits. Each delay-only fit passed the same-seed B physical prerequisite; each joint-delay group passed its same-seed delay-only prerequisite before execution.

| Selected-seed statistic | Median | Minimum | Maximum |
|---|---:|---:|---:|
| B mass relative error | 0.0103948% | 0.00856075% | 0.0172744% |
| B damping relative error | 0.0187225% | 0.00375350% | 0.0210152% |
| B held-out qpos RMSE (rad) | 0.002002920 | 0.001973256 | 0.002015245 |
| B held-out qvel RMSE (rad/s) | 0.019926075 | 0.019823297 | 0.020153696 |
| B held-out qpos improvement | 99.6935% | 99.6916% | 99.6980% |
| B held-out qvel improvement | 99.3237% | 99.3158% | 99.3271% |
| Joint-delay absolute error (microseconds) | 23.2718 | 4.30547 | 33.8387 |

Both B physical errors are within 10% in **5/5 seeds**, exceeding the required 4/5. Median held-out improvements exceed the unchanged 20% target for both sensors. Both individual held-out trajectories improve for every seed; neither has negative median improvement. Joint-delay error is below 0.002 s in **5/5 seeds**, and no B or joint-delay parameter hits its bound in any of the 30 fits. Delay-only's maximum absolute timing error is 31.1987 microseconds. On each seed, fixed-physics median absolute phase lag falls from 0.024 s to 0 s on the registered diagnostic grid.

Across B's three starts at any one seed, mass spread is at most 8.61e-10 kg and damping spread at most 2.34e-11 N*m*s/rad. Starting points affect native evaluation counts, but these starts found effectively the same optimum. The nominal S0 start is retained alongside the selected result; a tiny exact cost gap does not establish a material benefit from multiple starts. This experiment does not rule out difficult local minima in larger systems.

## Weak excitation did not fail

The 16-second single-step training trajectory also recovered both parameters for all five seeds. Its worst selected mass error was **0.0250012%**, and worst damping error **0.191435%**. Its median held-out RMSEs were **0.002002602 rad** and **0.019927192 rad/s**, very close to rich-excitation performance. Do not claim that a step was insufficient in this experiment.

The step did show weaker local sensitivity: selected scaled-Jacobian condition numbers were about 3.38 versus 1.34 for rich excitation, and damping errors varied more across seeds. Both Jacobians remained rank 2. A transient step can carry enough information when geometry, gain, initial state, and model family are known and only two physical parameters are free.

The comparison is not perfectly equal in every respect. Rich excitation uses two eight-second trajectories, two resets, and two 0.1 s zero-control prefixes; the step uses one sixteen-second trajectory, one reset, and one prefix. Both have 0.55 N*m active RMS. Sampled torque energy is 4.7795 `(N*m)^2*s` for rich training combined, versus 4.80975 for the step: approximately 0.63% more for the step. Active durations are 15.8 and 15.9 s, respectively, and their DC components differ. Residual sample counts are 15,764 and 15,882, so raw objective sums are not compared as if horizons were identical. The frozen inputs were preserved after observing the successful step outcome.

Delay excluded remains a separate ablation: selected mass errors are 1.81–1.84%, damping errors 5.99–6.02%, and held-out NRMSE 14.27–14.29 despite substantial improvement over nominal. Delay-aware fitting returns held-out NRMSE near 1. No held-out result chose a start or adjusted a bound, excitation, stopping option, or success target.

## Provenance and reproduction

The full invocation used clean commit `ec50b46b65344207915f33fc1db7ed47ddfc0ae7`. It completed in **57.173810 s**, executing **70 new fits** and verifying/reusing **12 immutable prior fits**: two A unit fits, three B seed-101 starts, and seven seed-101 delay-study fits. That is the measured cached full-study invocation, not a measured cold 82-optimization run. Original cache source commits and commands remain attached to each reused row. Static plotting/final-document assembly is outside this runner duration.

Environment: CPython 3.12.5; MuJoCo 3.15.0; macOS 26.6.2 / Darwin 25.6.0 ARM64; Apple M4 Pro, 12 CPU cores, 24 GB; one rollout worker and one numerical-library thread. Evidence/truth config hashes remain `f3e6deacceec9e6365b39e2c9067b9e3a8b71bab6a1c21f6f8e6ada4af824202` and `07c0aac32fe4147921572fdf3e8b134e49c991daa3468de27badb55ce9ef0f2f`.

The recorded command included `--output artifacts/full/20261013_evidence_v1`. Preserve existing evidence by leaving output unset for a fresh invocation:

```sh
.venv/bin/python scripts/run_experiment.py --config configs/evidence_v1.json --milestone 9
```

The runner validates frozen inputs/cache references, fitting prerequisites, shared objective equivalence, immutable training-only selections, and matrix accounting. Remote CI status is separate from these local evidence results. These findings are limited to authored synthetic trajectories and do not demonstrate real-robot robustness.
