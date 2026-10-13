"""Reduced smoke contracts; these tests do not assess full evidence targets."""

import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_smoke.py"
spec = importlib.util.spec_from_file_location("run_smoke", SCRIPT)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


def test_smoke_config_derives_reduced_copies_without_altering_registered_base():
    base_path = ROOT / "configs/evidence_v1.json"
    before = base_path.read_bytes()
    raw, study, truth = smoke.load_smoke_config(ROOT / "configs/smoke_v1.json")
    assert raw["scope"] == "verification_only_not_registered_evidence"
    assert study.duration_s == 1.2
    assert study.evaluation_window_s == (.06, 1.14)
    assert study.trajectories["heldout_pulse"].settings["pulses"][0]["start_s"] == .18
    assert study.trajectories["heldout_pulse"].settings["pulses"][-1]["end_s"] == 1.08
    assert study.nominal.mass_kg == .8 and study.nominal.damping_nm_s_per_rad == .04
    assert truth.model.mass_kg == 1.2 and truth.model.damping_nm_s_per_rad == .15
    assert base_path.read_bytes() == before


@pytest.mark.parametrize("update,error", [
    ({"base_evidence_sha256": "0"*64}, "hash differs"),
    ({"scope": "full_evidence"}, "verification-only"),
    ({"duration_s": .01}, "safe margins"),
    ({"duration_s": 1.201}, "timestep grid"),
    ({"noise_seed": True}, "integer seed"),
    ({"start_id": "S1"}, "one nominal"),
    ({"heldout_pulse_override": [{"start_s": .1, "end_s": 1.3, "amplitude_nm": .5}]}, "within duration"),
    ({"heldout_pulse_override": [{"start_s": .1, "end_s": .2, "amplitude_nm": 2}]}, "control bounds"),
])
def test_invalid_smoke_config_is_rejected(tmp_path, update, error):
    config = json.loads((ROOT/"configs/smoke_v1.json").read_text())
    config.update(update)
    path = tmp_path/"bad.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match=error):
        smoke.load_smoke_config(path)


@pytest.mark.parametrize("contents,error", [
    ('{"scope": "a", "scope": "b"}', "duplicate"),
    ('{"duration_s": NaN}', "non-finite"),
])
def test_smoke_json_requires_finite_unique_fields(tmp_path, contents, error):
    path = tmp_path/"invalid.json"
    path.write_text(contents)
    with pytest.raises(ValueError, match=error):
        smoke.load_smoke_config(path)


def test_ci_pins_actions_python_uv_and_always_retains_evidence():
    workflow = (ROOT/".github/workflows/ci.yml").read_text()
    assert workflow.count("uses:") == 3
    assert len(re.findall(r"uses: [\w/-]+@[0-9a-f]{40}", workflow)) == 3
    assert 'python-version: "3.12.5"' in workflow
    assert "uv==0.12.10" in workflow and "uv sync --locked --python 3.12.5" in workflow
    assert "scripts/run_smoke.py" in workflow and "if: always()" in workflow
    assert "path: artifacts/ci/" in workflow


def test_actual_one_fit_smoke_saves_finite_evidence_and_rejects_overwrite():
    # Provenance snapshots are only allowed under Git-ignored artifacts/.
    parent = ROOT/"artifacts/test_smoke_runner"
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="test_", dir=parent))
    output = temporary/"run"
    try:
        command = [sys.executable, str(SCRIPT), "--config", "configs/smoke_v1.json",
                   "--output", str(output)]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
        record = json.loads((output/"run.json").read_text())
        assert record["status"] == "passed"
        assert record["registered_evidence_fits"] == 0 and record["verification_fit_count"] == 1
        assert record["scope"] == "verification_only_not_registered_evidence"
        assert all(record["checks"].values())
        assert record["provenance"]["config"]["content"]["duration_s"] == 1.2
        assert record["provenance"]["mujoco_version"] == "3.15.0"
        assert len(record["provenance"]["git"]["commit"]) == 40
        if record["provenance"]["git"]["dirty"]:
            assert (output/"source_snapshot/manifest.json").exists()
        assert set(record["metrics"]) == {"nominal", "identified", "oracle"}
        assert record["metrics"]["identified"]["heldout"]["trajectory_count"] == 2
        assert record["official_outputs"]["videos_generated"] is False
        assert (output/"official/report.html").exists()
        with np.load(output/"residuals.npz", allow_pickle=False) as arrays:
            assert np.isfinite(arrays["initial"]).all() and np.isfinite(arrays["final"]).all()
        json.dumps(record, allow_nan=False)
        before = (output/"run.json").read_bytes()
        repeated = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=60)
        assert repeated.returncode != 0 and "Output already exists" in repeated.stderr
        assert (output/"run.json").read_bytes() == before
    finally:
        shutil.rmtree(temporary)
