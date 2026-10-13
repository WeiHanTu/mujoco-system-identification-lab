#!/usr/bin/env python3
"""Generate a compact results index and fixed-baseline table without rerunning fits."""

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default="docs/results/summary.json")
    parser.add_argument("--output", help="Fresh ignored producer directory; omit for UTC timestamp")
    args = parser.parse_args()
    from mujoco_sysid_lab.overview_reporting import generate_overview
    output = args.output or "artifacts/overview/"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    executable = Path(sys.executable).absolute()
    executable_text = executable.relative_to(ROOT).as_posix() if executable.is_relative_to(ROOT) else str(executable)
    try:
        result = generate_overview(ROOT, ROOT/args.summary, ROOT/output,
                                   [executable_text, "scripts/summarize_results.py", *sys.argv[1:]])
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps({"status": result["status"], "overview_bytes": result["overview"]["bytes"],
                      "registered_fit_count": result["registered_fit_count"],
                      "baseline_parameter_rows": result["baseline_parameters"]["row_count"],
                      "no_optimization_performed": True, "source_bytes_unchanged": True}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
