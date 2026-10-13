"""Regression checks for retaining immutable experiment evidence."""

from pathlib import Path
import runpy
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]


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


def test_existing_output_is_a_clear_cli_error_and_preserves_evidence():
    (ROOT / "artifacts").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT / "artifacts", prefix="pytest-cli-") as directory:
        output = Path(directory)
        evidence = output / "run.json"
        original = b'{"status":"retained"}\n'
        evidence.write_bytes(original)
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "verify_milestone1.py"),
             "--output", str(output)],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        assert result.returncode == 2
        assert "Omit --output" in result.stderr
        assert "already exists" in result.stderr
        assert "Traceback" not in result.stderr
        assert evidence.read_bytes() == original
        assert list(output.iterdir()) == [evidence]


def test_default_output_creates_fresh_directories_without_reusing_evidence():
    choose_output = runpy.run_path(str(ROOT / "scripts" / "verify_milestone1.py"))["_output_directory"]
    first = choose_output(None)
    first.mkdir(parents=True, exist_ok=False)
    try:
        second = choose_output(None)
        assert second != first
        assert not second.exists()
        assert second.parent == ROOT / "artifacts" / "milestone1"
    finally:
        first.rmdir()
