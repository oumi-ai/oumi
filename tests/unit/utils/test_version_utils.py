from unittest.mock import patch

import pytest

from oumi.utils.version_utils import is_dev_build


@pytest.fixture
def mock_version():
    with patch("oumi.utils.version_utils.version") as version_mock:
        yield version_mock


def test_is_dev_build_success(mock_version):
    mock_version.return_value = "0.1.0.dev0"
    assert is_dev_build()


def test_is_dev_build_failure(mock_version):
    mock_version.return_value = "0.1.0"
    assert not is_dev_build()
