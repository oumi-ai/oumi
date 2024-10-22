from pathlib import Path

import pytest


@pytest.fixture
def root_testdata_dir():
    return Path(__file__).parent / "testdata"
