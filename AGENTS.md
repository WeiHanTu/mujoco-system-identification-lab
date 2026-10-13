# Project execution contract

Project: `mujoco-system-identification-lab`.
Remote supplied by the user: `https://github.com/WeiHanTu/mujoco-system-identification-lab.git` (local origin).
Status: Milestone 1 complete and verified; stop before Milestone 2 until continuation.
Planning date: 2026-10-12, America/Los_Angeles.

## Working style and authorization

- Be direct and ruthless. Do not sugarcoat problems or agree for the sake of it.
- Read this file, `intend.md`, `spec.md`, and `plan.md` before making changes.
- This is an independent project under `/Users/weihantu/mujoco-lab/mujoco-system-identification-lab`. Never add its implementation to `sensor-fusion-replay-cpp`.
- The user approved the plan on 2026-10-12. The current authorization covers Milestone 1: pinned environment, installed API verification, minimal model/simulation, tests, and provenance. Later milestones require continuation under the checkpoint rule below.
- After each numbered milestone, stop, verify the stated gate, report evidence and failures, and wait for continuation unless the user explicitly authorizes several milestones together. This checkpoint rule comes from the user's execution-order instruction.
- The user supplied the existing GitHub repository and authorized connecting it as local origin. No upload, publication, or messages to other people are authorized by the planning request or by supplying the remote alone.

## Scientific integrity

- Describe the work as a reproducible synthetic dynamics system-identification study. Never describe it as real-robot validation, HIL, actuator calibration, production deployment, or completed sim-to-real transfer.
- Use only authored synthetic MJCF and data. Keep interview notes local and Git-ignored.
- Use MuJoCo's official SysID optimizer and pipeline. Do not substitute a separate optimizer to manufacture a successful result.
- Pin released MuJoCo 3.15.0 and all direct dependencies. Lock transitive dependencies. Never depend on Git main.
- Verify imports, exported symbols, installed signatures, and actual residual semantics. Report documentation/package mismatches before choosing supported hooks or writing any adapter.
- Hold actuator gain, geometry, COM, armature, gravity, and initial state fixed in the core experiment. Mass changes must scale rotational inertia consistently.
- Keep the ground-truth configuration out of the joint-estimator interface. Evaluation may read truth. The two explicitly labeled oracle-assisted single-parameter unit checks intentionally stamp the other true physical parameter into the spec; that controlled exception does not apply to joint recovery or delay fits.
- Train/test split units are complete trajectory IDs. Do not pass held-out controls, measurements, or truth to optimization, start selection, or hyperparameter tuning.
- Freeze configuration, seeds, starting values, bounds, metrics, stopping policy, and acceptance targets before the first identification evidence run. Changes require a new version and an explanation; retain earlier runs and failures.
- Preserve every initialization, seed, failure, bound hit, and non-convergence. Select among starts using training objective only.
- Distinguish implementation defects from disappointing scientific results. Fix correctness defects; do not lower thresholds or tune on held-out trajectories.
- Model mismatch can yield effective, biased parameters. This is not by itself proof of structural non-identifiability, and does not by itself rule out optimizer failure.

## Evidence and verification

- Every retained run must have exact config and its hash, seeds, command/arguments, MuJoCo/Python/package versions, OS/CPU, thread settings, Git commit, and dirty state. A dirty run also needs source hashes and its patch/untracked-source snapshot.
- Evidence runs require a real Git commit. Do not invent a commit identifier or label a dirty run clean.
- Raw data, parameter arrays, optimization logs, and machine-readable metrics go under ignored `artifacts/`. Curated public results must link to run IDs and hashes; an ignored directory alone is not durable evidence.
- Provide an explicit local archive command for complete synthetic evidence. Publication of that archive remains a separate user action.
- JSON must be strict JSON: reject NaN/Inf; represent unavailable diagnostics as null with a reason. Report singular Jacobians using rank/singular values and a status rather than non-finite numbers.
- All three fitting-model baselines use the same timestamps, initialization, observation transform, evaluation window, and metric code.
- Verify deterministic generation, safe timing/delay boundaries, split integrity, finite trajectories, physical mass/inertia modifiers, and truth-only mismatch isolation before scientific interpretation.
- Tests and CI must cover the meaningful contracts in `spec.md`; never report checks that were not executed.
- Start with CPU rollout, one rollout worker, and one numerical-library thread. Change concurrency only after recording profiling and reproducibility evidence.

## Before Git publication

Create `.gitignore` before generating artifacts or interview material. Ignore at least `.venv/`, caches, `artifacts/`, and `private/`. Audit tracked files and staged content for interview notes, credentials, machine identifiers, and unintended data. Public provenance needs OS/CPU, not serial numbers, hostnames, or home-directory identity.
