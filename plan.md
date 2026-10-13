# Execution plan

Status: Milestone 1 is complete and verified; Milestone 2 is authorized and in progress. Stop before Milestone 3.

## Current checkpoint

Workspace and available instructions inspected. Four requested planning files created in an independent project directory. Milestone 1 installed MuJoCo 3.15.0, verified official SysID imports/contracts, and retained an audited deterministic simulation. Milestone 2 now implements the registered training/held-out synthetic dataset and shared baseline evaluation; identification has not started.

The user supplied `https://github.com/WeiHanTu/mujoco-system-identification-lab.git`. A read-only Git check found no advertised branches or tags; the existing local repository is connected to it as origin. The user subsequently approved the plan and authorized Milestone 1. No remote publication has been requested.

The proposed dependencies, experiment matrix, acceptance targets, and claim boundaries were shown before approval. Milestone 1 verified released imports/API contracts, deterministic CPU simulation, and provenance. Its local suite passed 86 tests, with upstream warnings retained. See docs/milestones/01_environment.md and docs/results/milestone1.json. The user authorized the next milestone; stop before Milestone 3 identification.

## Milestone gates

At each milestone, stop execution before the next stage, verify the evidence, and report the gate outcome. Implementation correctness gates must pass. Scientific failures are preserved and explained; they cannot be fixed by changing thresholds or using held-out data. If a dependency or timing API behaves differently from the proposal, report the mismatch and its concrete impact before implementation adapts to it.

### Milestone 1 — Environment and one deterministic simulation

After approval: initialize/use the separate local repository, create .gitignore and pinned pyproject/lockfile, make a project-local Python 3.12.5 environment, and install only the released locked dependencies. Save resolver/import output and the complete installed version set.

Run the required `import mujoco; from mujoco import sysid`, assert MuJoCo 3.15.0, verify required symbols, inspect installed signatures and source paths, and map the official pipeline to this version. Check supported optimizer options and report entry points. Explicitly resolve the residual normalization, frozen modifier, mass/inertia, and delay-window concerns listed in spec.md; never assume the conceptual pseudocode is executable.

Author the small MJCF and minimal config/loader/CPU simulation path. Compile MjSpec, verify one hinge/motor/two sensors and parameter/inertia values. Run the same short noiseless trajectory twice. Verify shape, timestamps, finite states, exact initial state, endpoint/control alignment, and reproducibility. Save raw arrays, config, commands, source/environment provenance, and a short milestone record.

Gate: imports/exports/signatures verified, a released official API is usable, deterministic simulation checks pass. CPU profile is recorded. If the installed API disagrees with release documentation, report it before any adapter. Stop and show evidence.

### Milestone 2 — Synthetic datasets and nominal plots

Implement versioned truth/nominal/excitation/noise configs and deterministic dataset generation. Generate matched truth traces for the registered training and held-out IDs, plus nominal predictions on the same controls. Metadata records all physical/timing/noise choices. Keep truth separate from the estimator view.

Plot controls, clean/measured signals, and nominal errors; inspect the plots. Check amplitude/RMS/energy, trajectory horizons, sensor support, and absence of NaN/Inf. Confirm split integrity and hashes. No optimization occurs in this milestone.

Finalize the accepted configuration and register its hash, exact starts/bounds/metrics/criteria and optimizer options before Milestone 3. Any pre-evidence change must be explicit and justified by correctness or scope, not a wished-for recovery result. Do not create later evidence as if it used earlier settings.

Gate: deterministic dataset contracts and split tests pass; timing/inertia/noise generation audited; all constants are frozen before identification. Stop and show evidence.

### Milestone 3 — Independent mass and damping recovery

Build official named TimeSeries, explicit initial state, ModelSequences, noise-weighted residual, optimizer, and save/report path. Stamp the fixed physical parameter into the spec; do not rely on a frozen physical modifier. Check the residual against a direct known-noise calculation.

Run noiseless sanity checks and the two registered seed-101 single-parameter evidence fits. Save objective traces, recovered values, bounds/status/evaluations/runtime, and official reporting artifacts.

Gate: mass/inertia and damping modifiers behave correctly, official pipeline works, no truth/held-out leakage, and noiseless unit recovery succeeds within a declared numerical tolerance (proposed relative error <=0.1%). If noisy recovery is poor, retain it and inspect sensitivity rather than changing acceptance targets. Stop and show evidence.

### Milestone 4 — Joint physical recovery

Run all three matched seed-101 joint mass+damping starts with the frozen objective/options. Preserve the nominal-start result and choose the winner using training objective alone. Record its immutable fit ID for the later delay-only stage. Evaluate training predictions for nominal, identified, and oracle. Inspect residuals and scaled sensitivity diagnostics if accessible.

Gate: shared metrics and recorded optimizer objective agree with the registered conventions; results/diagnostics are finite or explicitly unavailable; recovered parameters and limitations are reported. Seed-101 delay-stage eligibility requires both winner physical errors <=10%; a miss is a retained scientific outcome, not permission to change bounds or acceptance criteria. No success criterion is silently adjusted. Stop and show evidence.

### Milestone 5 — Held-out prediction

Apply the already-fitted parameters to the untouched chirp/pulse trajectories without refitting. Generate the training/held-out comparisons, p95 errors, noise-normalized RMSE, improvement ratios, and latent-clean diagnostics. Verify all models use the same initialization and common observation window.

Gate: unseen controls never entered optimization or selection, held-out metrics are auditable, and both successes and failures are visible. Freeze no new settings based on these results. Stop and show evidence.

### Milestone 6 — Sensor delay

Generate target traces with a 0.024 s shared sensor delay and identical registered noise treatment. Verify positive-delay sign and fixed-window behavior with fixtures before identification. First fix physics to the immutable seed-101 B training-selected winner from Milestone 4 and fit delay only, provided both physical errors are <=10%. Compare with zero-delay prediction and the frozen phase-lag diagnostic. Record the source physical fit ID.

Run the registered seed-101 physical fits that assume zero delay. Only after delay-only timing correctness and absolute delay error <=0.002 s are verified, run seed-101 joint mass/damping/delay fitting. Distinguish phase correction from physical-parameter compensation and imperfect fixed physics from implementation defects. Use the registered starts; every candidate sees the same residual sample set.

Gate: delay boundary/sign tests pass; delay-only outcome and errors in timesteps are reported; joint delay fitting is either verified or explicitly deferred with its reason. Do not invent completed joint-delay results if the prerequisite fails. Stop and show evidence.

### Milestone 7 — Five seeds, three initializations, weak excitation

Complete the registered matched, weak-step, zero-delay-on-delayed, delay-only, and joint-delay cells for seeds 101–105 and their stated starts. Reuse completed immutable evidence fits rather than re-running them unnecessarily.

Choose each seed's multi-start winner solely by training objective. Save every row, including poor starts, failed convergence, bound hits, and not-run dependencies. Compare weak versus rich excitation with sampled RMS/energy/horizon/reset differences explicit. Aggregate median/min/max and criterion counts, with nominal-start behavior separately reported.

Gate: matrix accounting is complete or explicitly incomplete, all seed/start records retained, no held-out selection, and robustness targets assessed honestly. Stop and show evidence.

### Milestone 8 — Model mismatch

Generate targets with the registered smooth Coulomb-like torque, absent from fitting models. Verify truth-only force isolation. Fit only mass+damping across all five seeds and three starts. Evaluate held-out errors, parameter bias, sensitivity, and both the fitting-family oracle and generator-aware noise reference.

Explain what the actual evidence supports: reduced residual with biased effective parameters is consistent with misspecification. It does not prove structural non-identifiability or excuse uninvestigated optimization failure. If bias or improvement is small, publish that outcome instead of tuning the omitted torque to create a story.

Gate: model separation is tested, all 15 mismatch fits retained or marked failed, oracle meanings explicit, and claims match observed evidence. Stop and show evidence.

### Milestone 9 — Documentation, report, tests, evidence audit

Complete README, all five required figures, summary JSON, parameter CSV, official HTML report, full raw evidence archive command/hash, and a one-command CPU runner. README leads with measured held-out results, environment/runtime, and claim boundaries; plots and tables must be generated from stored results.

Run the contract/test suite and reduced CI smoke command. If remote CI is run, wait for its completed result before calling it passed. Inspect the final images and official report. Audit every public result back to config/dataset/source hashes, commit/dirty state, all seeds/starts, and acceptance criteria. Distinguish implemented, locally verified, and remote-CI-verified behavior.

Create ignored private bilingual interview notes, preserving the user's scripts only where completed results support them; cover all ten requested follow-ups. Audit tracked/staged files to prove those notes remain private. Do not publish or upload without an explicit user action.

Gate: requested artifacts exist, appropriate checks pass, retained failures are disclosed, provenance is complete, public claims are evidence-supported, and private notes are untracked. Stop and deliver the audit.

## Planned package layout and commands

Keep modules small: config validation, authored excitation/simulation, dataset IO, official SysID plumbing, shared evaluation, reporting/provenance, and experiment orchestration under src/mujoco_sysid_lab/. Avoid GPU/MJX, robotics middleware, a service layer, or an additional optimizer.

The following are proposed future command interfaces, not commands run during planning:

```sh
uv sync --locked --python 3.12.5
uv run --locked python scripts/generate_data.py --config configs/evidence_v1.json
uv run --locked python scripts/run_experiment.py --config configs/evidence_v1.json
uv run --locked pytest
uv run --locked python scripts/run_experiment.py --config configs/smoke_v1.json
```

The full runner generates/verifies cached datasets, runs all authorized registered cells, evaluates splits, and writes reports/plots/provenance with one command. Data generation is also exposed independently. It exits nonzero on invalid configs, contract failures, or incomplete execution, while writing partial evidence and reasons. Scientific target failures remain data and are clearly reported; an optional strict-criteria flag can make them a nonzero exit without disguising them as crashes.

Smoke uses a separate labeled config (short duration, seed 101, one joint start) and cannot overwrite full evidence. Its timing and threshold choices cannot be presented as the full study's success criteria. Target local runtime is <=60 s for smoke and <=15 minutes for full evidence; profiling, not confidence, decides whether the target was met.

## Risks to expose early

1. Released optional dependencies/API may differ from examples: installed-symbol/signature gate before estimator implementation.
2. Mass-only mutation may corrupt the intended physics: explicit inertia-scaling test.
3. Sensor/control offsets or unsafe delay interpolation may mimic timing error: alignment/sign/fixed-support tests.
4. Signal-amplitude normalization may accidentally change noise weighting: direct objective equivalence test.
5. A good fitted trajectory may have biased parameters: held-out and truth metrics, scaled sensitivity, and mismatch references.
6. Five seeds are descriptive evidence, not broad reliability proof: disclose denominators and all failures.
7. The matrix may miss the laptop runtime target: profile and report; do not silently reduce evidence coverage or introduce GPU complexity.
