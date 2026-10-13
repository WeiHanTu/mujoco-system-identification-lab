"""Invalid generation requests cannot overwrite retained evidence."""

from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def test_existing_output_is_rejected_without_touching_its_evidence():
    (ROOT / "artifacts").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT / "artifacts", prefix="pytest-dataset-cli-") as folder:
        output = Path(folder)
        original = output / "run.json"
        original.write_text('{"status":"retained"}\n')
        before = original.read_bytes()
        result = subprocess.run(
            [sys.executable, "scripts/generate_data.py", "--output", str(output)],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        assert result.returncode == 2
        assert "Omit --output" in result.stderr
        assert "Traceback" not in result.stderr
        assert original.read_bytes() == before
        assert list(output.iterdir()) == [original]


def test_unregistered_seed_is_a_usage_error():
    result = subprocess.run(
        [sys.executable, "scripts/generate_data.py", "--seed", "999"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 2
    assert "seed must be one of" in result.stderr
    assert "Traceback" not in result.stderr
