import subprocess
import sys


def test_manual_probe_is_safe_to_import_during_pytest_collection():
    completed = subprocess.run(
        [sys.executable, "-c", "import probe_test"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""


def test_model_smoke_module_has_no_import_side_effects():
    completed = subprocess.run(
        [sys.executable, "-c", "import test_import"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
