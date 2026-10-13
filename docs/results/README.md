# Synthetic study results

The retained study completed all **82 registered fits**, spanning five noise seeds and the registered starts. Matched-model median held-out position/velocity RMSE improved **99.69% / 99.32%** relative to the biased nominal model. Mass and damping recovered within the frozen 10% target for **5/5 seeds**. These findings concern authored synthetic observations only.

| Experiment | Median training normalized RMSE | Median held-out normalized RMSE | Interpretation |
|---|---:|---:|---|
| Matched mass + damping | 0.995 | 0.997 | Prediction reaches the injected noise scale. |
| Weak step | 1.005 | 0.998 | The step succeeds in this known-geometry, known-gain system. |
| Delay omitted | 11.757 | 14.282 | Timing error biases fitted physical parameters. |
| Delay only | 0.995 | 0.997 | Physics is fixed to the same-seed matched winner. |
| Joint delay + physics | 0.995 | 0.997 | Largest delay error is 0.03384 ms; target is 2 ms. |
| Omitted friction-like torque | 5.520 | 8.973 | Prediction improves while damping remains biased by 37.48%. |

Normalized RMSE combines position and velocity after dividing by their known noise scales. The mismatch residual remains above noise; its local confidence intervals do not establish coverage of true physics. Five deterministic seeds describe this experiment, not population reliability.

## Inspect the evidence

- [Compact overview](overview.json): every registered fit identity/status/start/parameter error and all 30 selected seed/experiment groups, plus frozen criteria, runtime, and median/minimum/maximum statistics. `parameter_columns` and `split_metric_columns` define the compact arrays; `initializations` and `aggregate_metrics` resolve shared records. Full source paths/hashes remain attached.
- [Full summary](summary.json): unchanged detailed result, including all initializations, trajectory p95 errors, local intervals, objective traces, and source provenance. Its SHA256 is recorded in the overview.
- [Parameter table](parameter_table.csv): all 172 free-parameter slots across the 82 fits.
- [Baseline parameter table](baseline_parameters.csv): nominal/oracle absolute and relative errors, bounds, fixed-value interpretation, and all seed/scenario rows. Fixed zero-delay assumptions are not optimizer bound pins.
- [Official selected matched report](../report.html) and [ablation reports](../reports/full): upstream HTML copied without modification. Its “nominal” observations mean measured signals; parameter-change errors compare against the starting reference.
- [Frozen registration](registration_v1.json), [verification](verification.json), [independent evidence audit](evidence_audit.json), and [local archive manifest](archive.json).
- [Checkpoint records](../milestones): historical Milestones 1–8, including preserved implementation failures and their corrections.

The recorded full-study command took **57.17 s** on the documented Apple M4 Pro CPU environment, executing **70 new optimizations and reusing 12 verified earlier fits**. This is not a measured cold 82-fit runtime. Final rendering, audit, and archive production have separate records. Remote CI has not been executed.

To view the official interactive HTML from the repository root:

```sh
python -m http.server 8000 --bind 127.0.0.1 --directory docs
```

Open <http://localhost:8000/report.html>. Upstream chart assets use a CDN, so fully offline interactive rendering is not verified. No hosted site or deployment is implied.

To regenerate only the compact overview from retained results:

```sh
uv run --locked python scripts/summarize_results.py --summary docs/results/summary.json
```

This performs no simulation, optimization, or start selection. It creates a fresh ignored producer directory, preserves the full summary bytes, and updates only the generated public overview.
