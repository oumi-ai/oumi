from pathlib import Path


def get_notebooks_dir() -> Path:
    return Path(__file__).parent.parent / "notebooks"
