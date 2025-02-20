import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import typer
from typer.testing import CliRunner

from oumi.cli.fetch import fetch

runner = CliRunner()


@pytest.fixture
def app():
    fake_app = typer.Typer()
    fake_app.command()(fetch)
    return fake_app


@pytest.fixture
def mock_fetch():
    with patch("oumi.cli.fetch.resolve_and_fetch_config") as fetch_mock:
        yield fetch_mock


def test_fetch_with_oumi_prefix_and_explicit_output_dir(app, mock_fetch):
    with tempfile.TemporaryDirectory() as temp_dir:
        # Given
        output_dir = Path(temp_dir)
        config_path = "oumi://configs/recipes/smollm/inference/135m_infer.yaml"
        expected_path = output_dir / "configs/recipes/smollm/inference/135m_infer.yaml"
        mock_fetch.return_value = expected_path
        # When
        result = runner.invoke(app, [config_path, "-o", str(output_dir)])

        # Then
        assert result.exit_code == 0
        mock_fetch.assert_called_once_with(config_path, output_dir, False)


def test_fetch_without_prefix_and_explicit_output_dir(app, mock_fetch):
    with tempfile.TemporaryDirectory() as temp_dir:
        # Given
        output_dir = Path(temp_dir)
        config_path = "configs/recipes/smollm/inference/135m_infer.yaml"
        expected_path = output_dir / "configs/recipes/smollm/inference/135m_infer.yaml"
        mock_fetch.return_value = expected_path
        # When
        result = runner.invoke(app, [config_path, "-o", str(output_dir)])

        # Then
        assert result.exit_code == 0
        mock_fetch.assert_called_once_with("oumi://" + config_path, output_dir, False)


def test_fetch_without_prefix_and_explicit_output_dir_force(app, mock_fetch):
    with tempfile.TemporaryDirectory() as temp_dir:
        # Given
        output_dir = Path(temp_dir)
        config_path = "configs/recipes/smollm/inference/135m_infer.yaml"
        expected_path = output_dir / "configs/recipes/smollm/inference/135m_infer.yaml"
        mock_fetch.return_value = expected_path
        # When
        result = runner.invoke(app, [config_path, "-o", str(output_dir), "-f"])

        # Then
        assert result.exit_code == 0
        mock_fetch.assert_called_once_with("oumi://" + config_path, output_dir, True)


def test_fetch_with_oumi_prefix_no_output(app, mock_fetch):
    # Given
    config_path = "oumi://configs/recipes/smollm/inference/135m_infer.yaml"
    expected_path = Path(config_path)
    mock_fetch.return_value = expected_path
    # When
    result = runner.invoke(app, [config_path])

    # Then
    assert result.exit_code == 0
    mock_fetch.assert_called_once_with(config_path, None, False)


def test_fetch_without_oumi_prefix_no_output(app, mock_fetch):
    # Given
    config_path = "configs/recipes/smollm/inference/135m_infer.yaml"
    expected_path = Path(config_path)
    mock_fetch.return_value = expected_path
    # When
    result = runner.invoke(app, [config_path])

    # Then
    assert result.exit_code == 0
    mock_fetch.assert_called_once_with("oumi://" + config_path, None, False)


def test_fetch_without_oumi_prefix_no_output_force(app, mock_fetch):
    # Given
    config_path = "configs/recipes/smollm/inference/135m_infer.yaml"
    expected_path = Path(config_path)
    mock_fetch.return_value = expected_path
    # When
    result = runner.invoke(app, [config_path, "-f"])

    # Then
    assert result.exit_code == 0
    mock_fetch.assert_called_once_with("oumi://" + config_path, None, True)
