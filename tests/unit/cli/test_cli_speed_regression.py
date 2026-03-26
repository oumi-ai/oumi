import subprocess
import sys


def _assert_module_not_imported(import_target: str, forbidden_module: str):
    """Verifies that importing import_target does not transitively import
    forbidden_module, by running the check in a clean subprocess."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import {import_target}; import sys; "
            f"assert '{forbidden_module}' not in sys.modules, "
            f"'{forbidden_module} was imported'",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"{forbidden_module} was imported by {import_target}. "
        f"This is a regression.\nstderr: {result.stderr}"
    )


def test_cli_speed_regression_no_torch_dependency():
    # Our CLI should have a relatively clean set of imports.
    # Importing torch is a sign that we are importing too much.
    _assert_module_not_imported("oumi.cli.main", "torch")


def test_cli_speed_regression_no_core_dependency():
    # Our CLI should have a relatively clean set of imports.
    # Importing oumi.core is a sign that we are importing too much.
    _assert_module_not_imported("oumi.cli.main", "oumi.core")
