import functools
from pathlib import Path


@functools.cache
def get_configs_dir() -> Path:
    return (Path(__file__).parent.parent / "configs").resolve()


@functools.cache
def get_testdata_dir() -> Path:
    return (Path(__file__).parent / "testdata").resolve()


@functools.cache
def get_notebooks_dir() -> Path:
    return (Path(__file__).parent.parent / "notebooks").resolve()
