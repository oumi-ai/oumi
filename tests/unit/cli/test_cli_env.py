import pytest
import typer
from typer.testing import CliRunner

from oumi.cli.env import env

runner = CliRunner()


#
# Fixtures
#
@pytest.fixture
def app():
    fake_app = typer.Typer()
    fake_app.command()(env)
    yield fake_app


def test_env_runs_without_exceptions(app):
    result = runner.invoke(app, [])
    assert result.exit_code == 0
    assert "Oumi environment information:" in result.stdout
