"""Test execution of notebooks."""

from pathlib import Path

import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor

from tests import get_notebooks_dir

# Notebooks whose tests we should skip, as they can't be run in an automated fashion.
# Reasons include that they're too slow, they start remote jobs, etc.
# An error will be thrown if a notebook in this set isn't found.
_NOTEBOOKS_TO_SKIP = {
    # Requires calling GPT-4 API
    "Oumi - Evaluation with AlpacaEval 2.0.ipynb",
    # Requires calling GPT-4 API
    "Oumi - Evaluation with MT Bench.ipynb",
    # Requires creating a remote instance
    "Oumi - Running Jobs Remotely.ipynb",
    # Requires creating a remote instance
    "Oumi - Using vLLM Engine for Inference.ipynb",
}


def get_notebooks():
    """Get all notebooks in the notebooks directory."""
    notebooks_dir = get_notebooks_dir()
    notebooks_to_skip = _NOTEBOOKS_TO_SKIP.copy()
    notebooks_to_test = []
    for notebook_path in notebooks_dir.glob("*.ipynb"):
        if notebook_path.name in notebooks_to_skip:
            notebooks_to_skip.remove(notebook_path.name)
        else:
            notebooks_to_test.append(notebook_path)
    if notebooks_to_skip:
        raise ValueError(f"Notebooks to skip not found: {notebooks_to_skip}")
    return notebooks_to_test


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
