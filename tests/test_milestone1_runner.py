"""Regression checks for retaining immutable experiment evidence."""

from pathlib import Path
import runpy


def test_setup_capture_never_hashes_live_runner_output(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "verify_milestone1.py"
    capture = runpy.run_path(str(script))["_setup_records"]
    setup = tmp_path / "artifacts" / "setup"
    setup.mkdir(parents=True)
    (setup / "commands.json").write_text("{}\n")
    active = setup / "milestone1-run.log"
    active.write_text("started\n")
    before = capture(tmp_path)
    active.write_text("started\ncompleted\n")
    assert capture(tmp_path) == before
    assert [record["path"] for record in before] == ["artifacts/setup/commands.json"]
