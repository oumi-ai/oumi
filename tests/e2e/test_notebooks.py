"""Test execution of notebooks."""

from pathlib import Path

import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor

from oumi.utils.io_utils import get_oumi_root_directory


def get_notebooks():
    """Get all notebooks in the notebooks directory."""
    notebooks_dir = get_oumi_root_directory().parent.parent / "notebooks"
    return list(notebooks_dir.glob("*.ipynb"))


@pytest.mark.parametrize("notebook_path", get_notebooks(), ids=lambda x: x.name)
def test_notebook_execution(notebook_path: Path):
    """Test that a notebook executes successfully."""
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=600)  # 10 minute timeout

    try:
        ep.preprocess(nb, {"metadata": {"path": notebook_path.parent}})
    except Exception as e:
        pytest.fail(f"Error executing notebook {notebook_path.name}: {str(e)}")
