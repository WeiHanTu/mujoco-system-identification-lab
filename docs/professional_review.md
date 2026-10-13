# Repository review after Milestone 9 — historical checkpoint

All nine scientific milestones passed before this review. This record describes the completed local review before the user's subsequent publication request. The work improves presentation, reproducibility, and failure handling while preserving the registered experiment and its retained results.

The README now leads with held-out prediction, scope, and a short reproduction command. Separate API and development guides explain interfaces, units, timing support, cache reuse, and evidence handling. A compact overview covers all 82 fits and 30 selected seed/scenario groups; a separate baseline table records nominal/oracle parameter errors. The detailed summary, original parameter table, five scientific figures, and official reports retain their original bytes.

The implementation records prepared/completed fits before evaluation, preserves runtime failures and interruptions, and marks incomplete checkpoints explicitly. Historical cache reuse verifies source provenance, frozen inputs, solver policy, and residual compatibility. `--fresh` disables reuse; explicit cache paths are optional. The archive utility accepts new full runs without requiring a historical Milestone 5 directory. Package metadata includes the README and repository links; direct pins and `uv.lock` remain unchanged.

## Final local verification

The clean source commit was `e1f663dfd61c976a0308c94def3538cd633e3420`. [professional_verification.json](results/professional_verification.json) is an exact copy of the retained producer record under `artifacts/verification/20261013_professional/`.

| Check | Recorded outcome |
|---|---|
| Full local test suite | 594 passed in 54.35 s |
| Upstream deprecation warnings | 739, retained in the test log |
| Short CPU SysID experiment | Passed |
| Milestone 6 cached replay | Passed; 10 retained fits, zero new optimizer calls |
| Wheel and source distribution | Built with pinned setuptools 80.9.0; package contents checked |
| Wheel Python modules | Byte-identical to the frozen source |
| Package privacy | No private notes, raw artifacts, or virtual environment payload |
| Frozen configuration | Unchanged |
| Remote CI | Configured; not executed |

The tests and replay took 58.36 s together. This is a verification runtime, separate from the earlier 57.17 s evidence run and 2.08 s public curation. The pinned upstream warnings concern Matplotlib/Pyparsing and a MuJoCo mass-modifier NumPy scalar conversion; they were not suppressed or resolved by changing dependency versions.

The public overview was produced without simulation or optimization from clean commit `0feff70`. The original evidence used clean commit `ec50b46`; its full run and detailed summary hashes remain unchanged. Local package builds validate packaging, not publication to a package index. Private bilingual interview notes remain Git-ignored. A license awaits an explicit owner choice.

## Durable local evidence

The subsequent archive is `artifacts/archives/20261013_professional_verified.tar.gz`, 166,899,487 bytes, SHA256 `0f3c822030bbc5a550c631641b7348127dd865b57d18ca7952beb18c0532f783`. All 1,599 regular member sizes and hashes passed verification after reopening. [professional_archive.json](results/professional_archive.json) retains its exact command, complete inventory, cached dependencies, and clean producer commit `0a2b6f0`.

It preserves the original scientific runs and failures, final verification logs, overview producer, package verification, and a verified Git source-history bundle through `0a2b6f0`. The bundle includes the historical source commits needed to reconstruct the retained runs. The later commit publishing this archive inventory is outside the bundle. The original Milestone 9 archive remains unchanged. At this checkpoint, archives and source history stayed local; no upload or repository publication had occurred.

This remains synthetic workflow verification. It establishes no real-robot validation, HIL, physical calibration, contact-rich manipulation, real sim-to-real transfer, production deployment, or general parameter identifiability.
