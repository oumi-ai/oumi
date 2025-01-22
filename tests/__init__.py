from pathlib import Path

from oumi.utils.io_utils import get_oumi_root_directory


def get_configs_dir() -> Path:
    result = (Path(__file__).parent.parent / "configs").resolve()
    print(f"get_configs_dir: {result}")
    return result


def get_testdata_dir() -> Path:
    result = (Path(__file__).parent / "testdata").resolve()

    old_result = (
        get_oumi_root_directory().parent.parent.resolve() / "tests" / "testdata"
    ).resolve()
    print(
        f"get_testdata_dir: NEW: {result} OLD: {old_result} EQ={result == old_result}"
    )
    return result


def get_notebooks_dir() -> Path:
    result = (Path(__file__).parent.parent / "notebooks").resolve()
    print(f"get_notebooks_dir: {result}")
    return result
