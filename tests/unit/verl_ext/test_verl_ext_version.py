import importlib.metadata

import pytest

import oumi.verl_grpo.config
from oumi_verl_ext import version
from oumi_verl_ext.version import (
    EXPECTED_VERL_VERSION,
    VerlVersionError,
    check_verl_version,
)


def test_expected_verl_version_is_oumi_s_pin():
    assert EXPECTED_VERL_VERSION == oumi.verl_grpo.config.VERL_VERSION


def test_check_verl_version_names_both_versions(monkeypatch):
    monkeypatch.setattr(version.importlib.metadata, "version", lambda name: "0.7.1")
    with pytest.raises(VerlVersionError, match=r"0\.7\.1.*0\.9\.1"):
        check_verl_version()


def test_check_verl_version_passes_for_the_pin(monkeypatch):
    monkeypatch.setattr(
        version.importlib.metadata, "version", lambda name: EXPECTED_VERL_VERSION
    )
    check_verl_version()


def test_check_verl_version_without_verl_says_not_installed(monkeypatch):
    def _missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(version.importlib.metadata, "version", _missing)
    with pytest.raises(VerlVersionError, match="not installed"):
        check_verl_version()
