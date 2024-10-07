import pytest
import typer
from typer.testing import CliRunner

from oumi.core.cli.cli_utils import (
    CONFIG_FLAGS,
    CONTEXT_ALLOW_EXTRA_ARGS,
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


def test_parse_extra_cli_args(app):
    # Verify that results are in the proper dot format.
    result = runner.invoke(app, ["--config", "some/path", "--allow_extra", "args"])
    expected_result = ["config=some/path", "allow_extra=args"]
    assert result.output.strip() == str(expected_result).strip()


def test_parse_extra_cli_args_empty(app):
    result = runner.invoke(app, [])
    expected_result = "[]"
    assert result.output.strip() == str(expected_result).strip()


def test_parse_extra_cli_args_fails_for_odd_args(app):
    # Verify that results are in the proper dot format.
    result = runner.invoke(app, ["--config", "some/path", "--odd"])
    assert "Extra arguments must be in" in result.output.strip()
