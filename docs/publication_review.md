# Public repository and delivery audit

The review followed a stranger's path: README → frozen inputs and results → source boundaries → clean checkout → tests, full experiment, independent audit, packaging, and Git delivery. Existing package boundaries were sufficient. The changes close specific contract and reproduction gaps; they add no dynamics capabilities or new optimizer.

| Actual gap | Correction and evidence |
|---|---|
| Matching dimensions admitted a slide joint, hidden actuator activation, or swapped sensor semantics. Missing named model nodes raised attribute errors. | Compilation, simulation, and estimator construction enforce the same narrow hinge/sensor contract. Negative fixtures cover each case. |
| Milestone 4 assumed its interpreter was inside the checkout; several tools recorded a guessed Python command. | Guarded path formatting preserves the actual executable in provenance and works with external interpreters. Regression fixtures check command records and truth-load order. |
| Independent auditing assumed cached unit fits duplicated `fitted_physics`. Fresh unit fits use the current estimation schema. | Reconstruct mechanics from hashed initial values and native free estimates; verify explicit records and the true fixed nuisance in oracle-assisted unit checks. Both fresh and historical 82-fit audits pass. |
| CI did not build the declared Python distributions; its old actions required a host-forced runtime upgrade. | CI builds wheel/sdist and retains them with tests and smoke output. Official Node 24 releases are pinned by full SHA. Experiments still require the Git checkout's scripts, model, and configs. |
| Planning examples contained an invalid smoke command and a nonexistent flag; historical checkpoints could look current. | Correct commands, distinguish historical records, identify primary CLI entry points, and document the source path and optional local-data test. |

## Local verification

The final source was clean commit `90433486b06361148f57641faec035d046556d7b`. An independent checkout with its own locked environment passed **618 tests, one skip, and 739 retained upstream deprecation warnings** in 83.17 s. The skipped test requires the author's historical ignored full study; portable synthetic fixtures run in CI. CPU smoke and wheel/source-distribution builds passed. Packaged Python modules match the checkout bytes.

A separate clean checkout of `8dc6942` completed all **82 registered fits with zero reuse**. The study took **81.62 s**; its full command, including curation, took **85.03 s**. All frozen criteria passed. The later `9043348` change fixes the auditor and its tests; it changes no dynamics or estimator implementation. No optimization was repeated to repair reporting.

Using the final clean auditor, both that fresh run and the original `ec50b46` run passed independent checks of **82 fits, 1,425 arrays, and 574 official files** each, with zero optimizer calls. The original 14 scientific artifact hashes, registration, configuration, lockfile, and pins remain unchanged. The README's scientific numbers continue to reference the original retained study.

[publication_verification.json](results/publication_verification.json) is the checked receipt for commands, inputs, versions, CPU/OS, commits, logs, package hashes, and audit records. It preserves links to the failed wrapper assertion and the discovered cached-schema assumption. Its receipt curation corrects a per-test warning-count parsing error and clarifies optimization-count scope. Failed records and raw runs remain intact.

## Review and publication boundaries

The [initial remote run](https://github.com/WeiHanTu/mujoco-system-identification-lab/actions/runs/38378517542) passed on `9043348`: contract tests, wheel/sdist, smoke, and artifact upload. [ci_verification.json](results/ci_verification.json) records its exact commit, Linux environment, test log, and smoke provenance. Its Node 20 runtime annotation led to release-pinned Node 24 updates: [checkout v7.0.1](https://github.com/actions/checkout/releases/tag/v7.0.1), [setup-python v7.0.0](https://github.com/actions/setup-python/releases/tag/v7.0.0), and [upload-artifact v7.0.2](https://github.com/actions/upload-artifact/releases/tag/v7.0.2). Subsequent CI results retain separate run identities.

The source path is documented in [development.md](development.md); [api.md](api.md) states shapes, units, supported mechanics, estimation/evaluation separation, and failures. Begin numerical review with the compact overview and parameter tables, then follow hashes into full results and native evidence. The supported full CLI generates synthetic inputs, seals selection by training cost, evaluates complete held-out controls, and curates reports.

Tracked files and reachable history were checked for credentials, private interview material, raw artifacts, and runtime files. Private notes and raw archives remain ignored. Existing Git authorship and historical source identifiers are preserved because evidence references those commits. No history rewrite or claim of anonymous history is made. Licensing remains undecided; no license was added.

This is synthetic workflow verification. It demonstrates no real-robot validation, HIL, physical actuator calibration, contact-rich manipulation, real sim-to-real transfer, production deployment, or general parameter identifiability.
