import importlib.metadata

import pytest

import oumi.verl_grpo.config
from oumi_verl_ext import version
from oumi_verl_ext.version import (
    EXPECTED_VERL_VERSION,
    VerlVersionError,
    check_verl_version,
)


def _install_verl(monkeypatch, installed: str | None) -> None:
    def fake_version(name):
        if installed is None:
            raise importlib.metadata.PackageNotFoundError(name)
        return installed

    monkeypatch.setattr(version.importlib.metadata, "version", fake_version)


def test_expected_verl_version_is_oumi_s_pin(monkeypatch):
    assert EXPECTED_VERL_VERSION == oumi.verl_grpo.config.VERL_VERSION
    _install_verl(monkeypatch, EXPECTED_VERL_VERSION)
    check_verl_version()


@pytest.mark.parametrize(
    ("installed", "match"), [("0.7.1", r"0\.7\.1.*0\.9\.1"), (None, "not installed")]
)
def test_check_verl_version_refuses_a_wrong_or_missing_verl(
    monkeypatch, installed, match
):
    _install_verl(monkeypatch, installed)
    with pytest.raises(VerlVersionError, match=match):
        check_verl_version()
