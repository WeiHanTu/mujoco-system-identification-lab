#!/usr/bin/env python3
"""Curate final synthetic study summaries, scientific plots, and official HTML."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["MPLBACKEND"] = "Agg"
os.environ["MPLCONFIGDIR"] = str(ROOT/"artifacts/matplotlib")
os.environ["XDG_CACHE_HOME"] = str(ROOT/"artifacts/cache")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Completed raw full-run directory or run.json")
    parser.add_argument("--output", help="New ignored rendering directory; omit for a timestamp")
    args = parser.parse_args()
    from mujoco_sysid_lab.final_reporting import curate_results
    default = "artifacts/curation/"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    executable = Path(sys.executable).absolute()
    argv0 = executable.relative_to(ROOT).as_posix() if executable.is_relative_to(ROOT) else str(executable)
    try:
        result = curate_results(ROOT, ROOT/args.run, ROOT/(args.output or default),
                                [argv0, "scripts/curate_results.py", *sys.argv[1:]])
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps({"status": result["status"], "public_artifact_count": len(result["public_artifacts"]),
                      "no_optimization_performed": True}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
