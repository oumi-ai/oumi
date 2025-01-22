from pathlib import Path


def get_configs_dir() -> Path:
    result = (Path(__file__).parent.parent / "configs").resolve()
    print(f"get_configs_dir: {result}")
    return result


def get_testdata_dir() -> Path:
    result = (Path(__file__).parent / "testdata").resolve()
    print(f"get_testdata_dir: {result}")
    return result


def get_notebooks_dir() -> Path:
    result = (Path(__file__).parent.parent / "notebooks").resolve()
    print(f"get_notebooks_dir: {result}")
    return result
