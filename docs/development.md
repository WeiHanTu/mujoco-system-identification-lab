# Development and reproduction

Read [AGENTS.md](../AGENTS.md), [intend.md](../intend.md), [spec.md](../spec.md), and [plan.md](../plan.md) before changing the experiment. Frozen v1 configuration, bounds, seeds, starts, noise, solver options, and thresholds must not change to improve outcomes. A different scientific experiment requires a new registered version and an explanation.

## Workflow

```sh
uv sync --locked --python 3.12.5
uv run --locked pytest -q
uv run --locked python scripts/run_smoke.py --config configs/smoke_v1.json
uv run --locked python scripts/run_experiment.py --config configs/evidence_v1.json
```

Use uv 0.12.10. Python and all direct dependencies are pinned; `uv.lock` locks transitive versions/hashes. The separate smoke configuration shortens the horizon and uses one seed/start; it contributes zero registered evidence fits. CI runs tests, wheel/source-distribution builds, and smoke on Ubuntu, then retains logs, packages, and smoke evidence for 14 days. Local verification does not prove remote CI or cross-platform bitwise reproducibility.

Each run gets a fresh ignored `artifacts/` directory. Leave `--output` unset for timestamped outputs; explicit existing paths are rejected. The full command updates tracked public summaries, plots, and official reports from retained results. Review those generated diffs alongside their source/producer hashes. Running a verification command alone does not overwrite the frozen scientific configuration.

`--fresh` disables historical fit reuse. Optional cache-run arguments are documented by `scripts/run_experiment.py --help`. Automatic legacy checkpoint paths support reproducing the retained local study; they are optional and absent in a clean clone. Cache identities include training inputs, model/config, starts, physical constants, solver options, and retained source provenance. Held-out scores never select a start or determine reuse.

## Module map

Follow `scripts/run_experiment.py` into `study_runner.run_study` to review the full matrix. Dataset loaders return restricted training inputs; `identification.build_problem` connects them to official SysID. Selection is sealed from training cost before `full_evaluation` reads truth and held-out signals. `final_reporting` consumes saved results to produce public evidence. The separate `joint_experiment` and milestone scripts preserve earlier checkpoint workflows; they are not alternative optimizers.

| Module | Responsibility |
|---|---|
| `config.py`, `experiment_config.py` | Strict mechanics and frozen experiment validation. |
| `excitation.py`, `simulation.py` | Authored controls and fresh deterministic CPU state/sensor simulation. |
| `dataset.py`, `advanced_data.py` | Versioned NPZ/JSON, paired noise, delay/mismatch generation, restricted training loaders. |
| `api_check.py`, `identification.py` | Installed API/timing checks and official bounded SysID problem. |
| `joint_identification.py`, `study_runner.py` | Training-only estimation/selection and registered matrix orchestration. |
| `evaluation.py`, `full_evaluation.py` | Shared metrics and separate post-selection full-horizon evaluation. |
| `fit_reporting.py`, `final_reporting.py`, `overview_reporting.py` | Official native/HTML saves, scientific figures, full/compact public results. |
| `provenance.py` | Exact config/source/environment hashes and dirty-source snapshots. |

Supported shapes, units, interfaces, and failure behavior are in [api.md](api.md). The tests focus on deterministic generation, model/timing integrity, train/held-out isolation, bounds, objective equivalence, native reporting, failure retention, and evidence schemas.

The supported experiment interface requires the Git checkout: the wheel contains Python helpers, while scripts, MJCF, and frozen configuration live in the repository. Building a wheel does not make it a standalone experiment application. One integration test also checks the author's retained full study when those ignored inputs exist; a clean checkout skips that test with an explicit reason and exercises synthetic contract fixtures instead. Full local counts and clean-checkout/CI counts can therefore differ by one.

## Audit and archive

Replace `<run>` with an actual generated directory. Output paths must be new:

```sh
uv run --locked python scripts/audit_evidence.py \
  --run artifacts/<run> --output artifacts/audits/<new-name>.json

uv run --locked python scripts/archive_evidence.py \
  --run artifacts/<run> \
  --output artifacts/archives/<new-name>.tar.gz \
  --manifest artifacts/archives/<new-name>.json
```

The independent audit replays saved full-precision models and computes residuals/metrics with direct NumPy formulas; it never optimizes. Source reconstruction requires the recorded Git history and dirty snapshots. The archive follows cached fit dependencies, preserves raw inputs/native reports/provenance, and checks every regular member's size/SHA256 after reopening. Historical checkpoints and additional audit/rendering records are included explicitly with repeated `--include` arguments. Archiving a new full run does not require the author's historical Milestone 5 directory.

Archive hashes detect changed bytes; they are not an independent replication. Native pickles are local authored artifacts and are hash-verified before loading. A third party should reproduce from trusted source/data rather than deserialize an arbitrary downloaded pickle.

## Evidence policy

Preserve failures, all starts/seeds, native convergence flags, and unavailable diagnostics. Finite nonconverged candidates remain eligible for training-only selection. Report scientific target failures separately from execution/contract failures. Keep generator truth and omitted forces out of the core estimator; only explicitly labeled unit checks use a true nuisance parameter.

Plots and tables come from saved results. Do not edit numerical cells or images to manufacture a cleaner outcome. Full reports remain byte-identical to official generated artifacts. Private interview material, virtual environments, caches, and raw runs stay Git-ignored. This study demonstrates a synthetic workflow, not real-robot validation, HIL, or deployment.
