import os
from datetime import datetime
from pathlib import Path


def get_e2e_test_output_dir(test_name: str, tmp_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if os.environ.get("OUMI_E2E_TESTS_OUTPUT_DIR"):
        output_base = Path(os.environ["OUMI_E2E_TESTS_OUTPUT_DIR"])
    else:
        output_base = tmp_path / "e2e_tests"

    return output_base / (
        f"{timestamp}"  # Only timestamp if test_name is in output_base already.
        if f"/{test_name}/" in str(output_base)
        else f"{timestamp}_{test_name}"
    )


def is_file_not_empty(file_path: Path) -> bool:
    """Check if a file is not empty."""
    return file_path.stat().st_size > 0
