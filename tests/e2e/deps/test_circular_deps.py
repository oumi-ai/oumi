import glob
import os
import re
from multiprocessing import Process, set_start_method
from pathlib import Path
from typing import Optional

import pytest


def _backtrack_on_path(path, n):
    """Goes up n directories in the current path."""
    output_path = path
    for _ in range(n):
        output_path = os.path.dirname(output_path)
    return output_path


def _get_oumi_path_recursively(path: Path) -> str:
    """Recursively goes up the path until it finds the oumi dir."""
    if len(path.name) == 0:
        raise FileNotFoundError("Could not find oumi dir.")
    if path.name == "oumi":
        return path.name
    return f"{_get_oumi_path_recursively(path.parent)}.{path.stem}"


def _get_all_py_paths(exclude_patterns: Optional[set[str]]) -> list[str]:
    """Recursively returns all py files in the /src/oumi/ dir of the repo."""
    path_to_current_file = os.path.realpath(__file__)
    repo_root = _backtrack_on_path(path_to_current_file, 4)
    py_pattern = str(Path(repo_root) / "src" / "oumi" / "**" / "*.py")
    all_py_files = glob.glob(py_pattern, recursive=True)
    exclude_files = []
    if exclude_patterns:
        for file in all_py_files:
            for exclude_pattern in exclude_patterns:
                if re.fullmatch(exclude_pattern, file):
                    exclude_files.append(file)
                    break
    all_py_files = [
        _get_oumi_path_recursively(Path(file))
        for file in all_py_files
        if file not in exclude_files
    ]
    assert len(all_py_files) > 0, "No py files found to parse."
    return all_py_files


all_paths = _get_all_py_paths(
    exclude_patterns={".*__init__.py", str(Path(".*") / "experimental" / ".*")},
)


def _load_module(module_path: str):
    """Loads a module from a path."""
    import importlib

    return importlib.import_module(module_path)


@pytest.mark.parametrize(
    "py_path",
    all_paths,
    ids=all_paths,
)
def test_load_py_files(py_path: str):
    # Load all python files in our src/oumi/ directory.
    # Circular dependencies will manifest as an ImportError.
    set_start_method("spawn", force=True)
    # Create a new process with a clean dependency tree.
    process = Process(target=_load_module, args=[py_path])
    process.start()
    process.join()
    assert process.exitcode == 0, (
        f"Error loading {py_path}. Exitcode: {process.exitcode}"
    )
