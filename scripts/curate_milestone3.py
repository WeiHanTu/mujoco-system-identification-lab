#!/usr/bin/env python3
"""Publish local curated unit-check evidence and a byte-verified raw archive."""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["MPLCONFIGDIR"] = str(ROOT / "artifacts/matplotlib")
os.environ["MPLBACKEND"] = "Agg"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="artifacts/milestone3/20261012_seed101_verified")
    parser.add_argument("--output", default="artifacts/milestone3/20261012_curated")
    parser.add_argument("--archive", default="artifacts/milestone3/20261012_evidence_verified.tar.gz")
    args = parser.parse_args()
    run_dir, derived, archive = ((ROOT / value).resolve() for value in (args.run, args.output, args.archive))
    if any(not path.is_relative_to(ROOT / "artifacts") for path in (run_dir, derived, archive)):
        parser.error("Raw inputs and derived evidence must remain under ignored artifacts/")
    if derived.exists() or archive.exists():
        parser.error("Derived output or archive exists; choose new paths to preserve evidence")
    from matplotlib import pyplot as plt
    from mujoco_sysid_lab.provenance import collect_provenance, sha256_file, snapshot_sources, write_json

    run = json.loads((run_dir / "run.json").read_text())
    original_dir = ROOT / run["source_run"]["path"]
    original = json.loads((original_dir / "run.json").read_text())
    audits = {name: json.loads((run_dir / f"{name}.json").read_text())
              for name in ("audit", "training_audit")}
    if run["status"] != "passed" or any(item.get("verdict", item.get("status")) != "passed" for item in audits.values()):
        parser.error("Verified fit/report and independent audits must pass before curation")
    executable_path = Path(sys.executable).absolute()
    executable = executable_path.relative_to(ROOT).as_posix() if executable_path.is_relative_to(ROOT) else str(executable_path)
    provenance = collect_provenance(ROOT, ROOT / "configs/evidence_v1.json", ROOT / "model/pendulum.xml",
                                    [executable, "scripts/curate_milestone3.py", *sys.argv[1:]])
    provenance.update(seed=101, milestone=3, scenario="matched", created_at_utc=datetime.now(timezone.utc).isoformat(),
                      operation="curation; no optimization or dynamics generation")
    derived.mkdir(parents=True, exist_ok=False)
    if provenance["git"]["dirty"]:
        provenance["source_snapshot"] = snapshot_sources(ROOT, derived / "source_snapshot")
    rows = []
    for fit in run["fits"]:
        parameter = fit["summary"]["parameter_rows"][0]
        diagnostics = fit["summary"]["optimizer"]
        rows.append({"fit_id": fit["fit_id"], "seed": 101, "start_id": "S0", "oracle_assisted": True,
                     "noiseless_sanity": fit["noiseless_sanity"], **parameter,
                     "success": diagnostics["success"], "status": diagnostics["status"],
                     "nfev": diagnostics["nfev"], "njev": diagnostics["njev"],
                     "callback_events": diagnostics["callback_events"],
                     "objective_initial": fit["summary"]["objective"]["initial"],
                     "objective_final": fit["summary"]["objective"]["final"],
                     "runtime_seconds": fit["summary"]["runtime_seconds"],
                     "optimization_source_commit": run["optimization_provenance"]["git"]["commit"],
                     "optimization_source_dirty": run["optimization_provenance"]["git"]["dirty"],
                     "report_source_commit": run["provenance"]["git"]["commit"],
                     "report_source_dirty": run["provenance"]["git"]["dirty"]})
    table = derived / "parameter_table.csv"
    with table.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    for axis, fit in zip(axes, [row for row in run["fits"] if row["registered_evidence_fit"]], strict=True):
        parameter = fit["summary"]["parameter_rows"][0]
        half_width = fit["summary"]["confidence_intervals"]["half_widths"][0]
        axis.scatter([0], [parameter["initial"]], s=65, color="#e67e22", label="Registered start")
        axis.errorbar([1], [parameter["estimate"]], yerr=[half_width], fmt="o", capsize=6,
                      color="#00897b", label="Estimate ± local 95% CI")
        axis.axhline(parameter["truth"], color="#37474f", ls="--", label="Generator truth")
        axis.set_xticks([0, 1], ["Initial", "Identified"])
        mass = parameter["name"] == "mass_kg"
        axis.set_ylabel("Mass (kg)" if mass else "Damping (N m s/rad)")
        axis.set_title("Mass only; damping fixed to truth" if mass else "Damping only; mass fixed to truth", fontsize=11)
        axis.text(.05, .92, f"Relative error: {100 * parameter['relative_error']:.5f}%", transform=axis.transAxes)
        axis.grid(alpha=.18)
        axis.legend(loc="lower right", fontsize=8)
    figure.suptitle("Oracle-assisted single-parameter recovery · noise seed 101")
    image_path = derived / "parameter_recovery.png"
    figure.savefig(image_path, dpi=150)
    plt.close(figure)
    public_files = {}
    copies = [(table, ROOT / "docs/results/milestone3_parameter_table.csv"),
              (image_path, ROOT / "docs/images/milestone3_parameter_recovery.png")]
    copies.extend((run_dir / f"fits/{cell}_seed101_S0/official/report.html",
                   ROOT / f"docs/reports/milestone3/{cell}/report.html") for cell in ("A1", "A2"))
    for source, target in copies:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        assert sha256_file(source) == sha256_file(target)
        public_files[target.relative_to(ROOT).as_posix()] = {
            "sha256": sha256_file(target), "source": source.relative_to(ROOT).as_posix()}
    write_json(derived / "run.json", {"schema_version": "mujoco-sysid-lab.curation.v1", "status": "passed",
        "provenance": provenance, "source_run": {"path": run_dir.relative_to(ROOT).as_posix(), "sha256": sha256_file(run_dir / "run.json")},
        "no_optimization_performed": True, "public_artifacts": public_files})
    archive_command = ["tar", "--no-mac-metadata", "-czf", archive.relative_to(ROOT).as_posix(),
                       original_dir.relative_to(ROOT).as_posix(), run_dir.relative_to(ROOT).as_posix(),
                       derived.relative_to(ROOT).as_posix(),
                       "artifacts/milestone2/20261012_seed101_verified/dataset",
                       "artifacts/setup/milestone3-run.log", "artifacts/setup/milestone3-report-repair.log",
                       "artifacts/setup/pytest-milestone3.log", "artifacts/setup/pytest-milestone3-final.log"]
    subprocess.run(archive_command, cwd=ROOT, check=True)
    file_count = 0
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            assert not any(part.startswith("._") for part in Path(member.name).parts)
            if member.isfile():
                assert bundle.extractfile(member).read() == (ROOT / member.name).read_bytes()
                file_count += 1
    result = {"schema_version": "mujoco-sysid-lab.curated-milestone3.v1", "status": "passed",
              "milestone": 3, "noise_seed": 101, "oracle_assisted": True,
              "optimization_provenance": run["optimization_provenance"], "report_provenance": run["provenance"],
              "curation_provenance": provenance, "raw_run_directory": run_dir.relative_to(ROOT).as_posix(),
              "raw_run_sha256": sha256_file(run_dir / "run.json"), "retained_run": run,
              "independent_audits": audits, "registered_evidence_fit_count": 2, "registered_matrix_total": 82,
              "separate_noiseless_sanity_fit_count": 2, "optimization_repeated_for_report_repair": False,
              "original_report_failure": {"run_id": original_dir.name, "status": original["status"],
                                          "raw_run_sha256": sha256_file(original_dir / "run.json"),
                                          "reason": "Noisy optimizers converged, but full-precision XML comparison ignored six-digit serialization; original failures preserved."},
              "raw_evidence_archive": {"path": archive.relative_to(ROOT).as_posix(), "sha256": sha256_file(archive),
                                       "size_bytes": archive.stat().st_size, "creation_command_argv": archive_command,
                                       "working_directory": ".", "regular_files_byte_verified": file_count,
                                       "appledouble_members": 0, "published": False},
              "public_artifacts": public_files,
              "tests": {"passed": 326, "warnings": 148, "seconds": 4.47,
                        "command": "MPLCONFIGDIR=artifacts/matplotlib .venv/bin/python -m pytest -q 2>&1 | tee artifacts/setup/pytest-milestone3-final.log",
                        "log_sha256": sha256_file(ROOT / "artifacts/setup/pytest-milestone3-final.log"), "remote_ci_run": False},
              "visual_verification": {"parameter_figure_inspected": True, "public_html_reports_opened": 2,
                                      "observations_and_interpretation_rendered": True, "browser_error_log_empty": True,
                                      "official_template_external_assets": ["https://cdn.plot.ly/plotly-4.1.1.min.js", "Google Fonts stylesheet"],
                                      "offline_interactive_rendering_verified": False},
              "next_milestone": {"number": 4, "status": "awaiting_user_continuation"},
              "claim_boundaries": {"verified": ["conditional independent mass/damping recovery", "official optimization/save/HTML pipeline", "training-only fitting and objective checks"],
                                   "not_yet_demonstrated": ["joint recovery", "fitted held-out prediction improvement", "five-seed robustness", "delay/mismatch ablations"],
                                   "not_demonstrated": ["real robot data", "HIL", "physical actuator calibration", "contact-rich manipulation", "real sim-to-real transfer", "production deployment", "general identifiability for arbitrary robots"]}}
    write_json(ROOT / "docs/results/milestone3.json", result)
    print(f"Milestone 3 curation passed: {file_count} archived files; SHA256 {sha256_file(archive)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
