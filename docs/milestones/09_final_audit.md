# Milestone 9 — Final evidence audit

All nine milestones are complete. The registered experiment retained 82 unique fits, all converged, with no parameter bound hits. Every frozen engineering target passed. The weak-step ablation succeeded; the mismatch residual remains above noise despite substantially improved prediction and biased damping. These findings are preserved in the final results.

| Verification | Recorded result |
|---|---|
| Local test suite | 541 passed, 739 upstream warnings, 23.73 s |
| CPU smoke | Passed; one shortened verification fit, zero registered evidence fits |
| Independent native/evaluation audit | 82 fits, 1,425 arrays, 574 official files verified |
| Maximum independently recomputed residual difference | 5.55e-14 |
| Public curation | 14 files byte-verified; 2.082 s; no optimization |
| Official report browser check | 95 SVG elements; no console errors; no horizontal overflow at 1280 px |
| Evidence archive | 1,532 regular members; every size and SHA256 verified |
| Private notes | Git-ignored and absent from tracked files |
| Remote CI | Configured; not run |

The clean-source verification command was `.venv/bin/python -m pytest -q`, followed by the explicit shortened smoke command. Its source commit was `b9635c1`, recorded with full hashes in [verification.json](../results/verification.json). The final curation used clean commit `756db3e`; its producer record is `artifacts/curation/20261013_final_clean_verified/run.json`. The 57.17 s experiment used clean commit `ec50b46`, ran 70 new fits, and reused 12 verified immutable earlier fits. These timings describe distinct operations; an uncached 82-fit execution was not timed.

The raw archive is `artifacts/archives/20261013_evidence_v1_verified.tar.gz`, 159,726,162 bytes, SHA256 `70f2c0dfc8a0a4d79ff2a5e6ec6b662d1451c68e95fc7547ce6fbc2760845a95`. Its exact creation command, complete member inventory, cached dependencies, retained earlier failures, configs, commands, versions, source provenance, and verification result are recorded in [archive.json](../results/archive.json). The archive remains local. Source reconstruction uses the repository's recorded Git history plus any retained dirty-source snapshot.

All five figures were inspected. The official HTML was checked in a browser with upstream CDN assets available; fully offline interactive rendering is not verified. The generated report's “nominal” observation label and starting-reference parameter-change columns are explained in the report and README. Full-precision MJB/native arrays are preserved alongside six-significant-digit XML.

The independent audit uses fresh CPU predictions from saved full-precision models and manual NumPy residual/metric calculations, without calling an optimizer. It checks original Git source bytes, all native vectors/Jacobians/objectives, training-only selections, fixed physics, delay prerequisites, phase diagnostics, bounds, and generator-aware references. [evidence_audit.json](../results/evidence_audit.json) records the checks and its separate audit provenance.

No acceptance threshold, frozen physical value, excitation, bound, seed, or solver option changed to improve outcomes. Synthetic verification does not demonstrate real robot data, HIL, actuator calibration, contact-rich manipulation, sim-to-real transfer, or production deployment.
