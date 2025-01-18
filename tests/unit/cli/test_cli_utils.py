import os
from unittest import mock

import pytest
import typer
from typer.testing import CliRunner

from oumi.cli.cli_utils import (
    CONFIG_FLAGS,
    CONTEXT_ALLOW_EXTRA_ARGS,
    LogLevel,
    configure_common_env_vars,
    parse_extra_cli_args,
)


def simple_command(ctx: typer.Context):
    print(str(parse_extra_cli_args(ctx)))


runner = CliRunner()


#
# Fixtures
#
@pytest.fixture
def app():
    fake_app = typer.Typer()
    fake_app.command(context_settings=CONTEXT_ALLOW_EXTRA_ARGS)(simple_command)
    yield fake_app


def test_config_flags():
    # Simple test to ensure that this constant isn't changed accidentally.
    assert CONFIG_FLAGS == ["--config", "-c"]


def test_context_allow_extra_args():
    # Simple test to ensure that this constant isn't changed accidentally.
    assert CONTEXT_ALLOW_EXTRA_ARGS == {
        "allow_extra_args": True,
        "ignore_unknown_options": True,
    }


def test_parse_extra_cli_args_space_separated(app):
    # Verify that results are in the proper dot format.
    result = runner.invoke(app, ["--config", "some/path", "--allow_extra", "args"])
    expected_result = ["config=some/path", "allow_extra=args"]
    assert result.output.strip() == str(expected_result).strip()


def test_parse_extra_cli_args_eq_separated(app):
    # Verify that results are in the proper dot format.
    result = runner.invoke(app, ["--config=some/path", "--allow_extra=args"])
    expected_result = ["config=some/path", "allow_extra=args"]
    assert result.output.strip() == str(expected_result).strip()


def test_parse_extra_cli_args_mixed(app):
    # Verify that results are in the proper dot format.
    result = runner.invoke(
        app, ["--config=some/path", "--foo ", " bar ", "--bazz = 12345 ", "--zz=XYZ"]
    )
    expected_result = ["config=some/path", "foo=bar", "bazz=12345", "zz=XYZ"]
    assert result.output.strip() == str(expected_result).strip()


def test_parse_extra_cli_args_empty(app):
    result = runner.invoke(app, [])
    expected_result = "[]"
    assert result.output.strip() == str(expected_result).strip()


def test_parse_extra_cli_args_fails_for_odd_args(app):
    # Verify that results are in the proper dot format.
    result = runner.invoke(app, ["--config", "some/path", "--odd"])
    output_str = result.output.strip()
    assert "Trailing argument has no value assigned" in output_str, f"{output_str}"


def test_valid_log_levels():
    # Verify that the log levels are valid.
    expected_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    supported_levels = set(LogLevel.__members__.keys())
    assert expected_levels == supported_levels


@mock.patch.dict(os.environ, {"FOO": "1"}, clear=True)
def test_configure_common_env_vars_empty():
    configure_common_env_vars()
    assert os.environ == {
        "FOO": "1",
        "ACCELERATE_LOG_LEVEL": "info",
        "TOKENIZERS_PARALLELISM": "false",
    }


@mock.patch.dict(
    os.environ,
    {
        "TOKENIZERS_PARALLELISM": "true",
        "FOO": "1",
    },
    clear=True,
)
def test_configure_common_env_vars_partially_preconfigured():
    configure_common_env_vars()
    assert os.environ == {
        "FOO": "1",
        "ACCELERATE_LOG_LEVEL": "info",
        "TOKENIZERS_PARALLELISM": "true",
    }


@mock.patch.dict(
    os.environ,
    {"TOKENIZERS_PARALLELISM": "true", "ACCELERATE_LOG_LEVEL": "debug"},
    clear=True,
)
def test_configure_common_env_vars_fully_preconfigured():
    configure_common_env_vars()
    assert os.environ == {
        "ACCELERATE_LOG_LEVEL": "debug",
        "TOKENIZERS_PARALLELISM": "true",
    }
