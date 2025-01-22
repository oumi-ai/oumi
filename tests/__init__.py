from pathlib import Path


def get_configs_dir() -> Path:
    return Path(__file__).parent.parent / "configs"


def get_testdata_dir() -> Path:
    return Path(__file__).parent / "testdata"


def get_notebooks_dir() -> Path:
    return Path(__file__).parent.parent / "notebooks"
