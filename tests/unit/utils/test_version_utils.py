from unittest.mock import patch

from oumi.utils.version_utils import is_dev_build


def test_is_dev_build_success():
    with patch("oumi.utils.version_utils.get_oumi_version", return_value="0.1.0.dev0"):
        assert is_dev_build()


def test_is_dev_build_failure():
    with patch("oumi.utils.version_utils.get_oumi_version", return_value="0.1.0"):
        assert not is_dev_build()


def test_is_dev_build_none():
    with patch("oumi.utils.version_utils.get_oumi_version", return_value=None):
        assert not is_dev_build()
