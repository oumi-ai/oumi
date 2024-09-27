import pytest

from oumi.utils.io_utils import get_oumi_root_directory


@pytest.mark.parametrize("filename", ["train.py", "evaluate.py", "launch.py"])
def test_get_oumi_root_directory(filename):
    root_dir = get_oumi_root_directory()
    file_path = root_dir / filename
    assert file_path.exists(), f"{file_path} does not exist in the root directory."
