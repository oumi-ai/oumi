import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

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
def mock_response():
    response = Mock()
    response.text = "key: value"
    response.raise_for_status.return_value = None
    return response


@pytest.fixture
def mock_requests(mock_response):
    with patch("oumi.cli.fetch.requests") as mock:
        mock.get.return_value = mock_response
        yield mock


def test_fetch_with_explicit_output_dir(app, mock_requests):
    with tempfile.TemporaryDirectory() as temp_dir:
        # Given
        output_dir = Path(temp_dir)
        config_path = "oumi://smollm/inference/135m_infer.yaml"
        expected_path = output_dir / "smollm/inference/135m_infer.yaml"

        # When
        result = runner.invoke(app, [config_path, "-o", str(output_dir)])

        # Then
        assert result.exit_code == 0
        mock_requests.get.assert_called_once()
        assert expected_path.exists()


def test_fetch_with_oumi_dir_env(app, mock_requests, monkeypatch):
    with tempfile.TemporaryDirectory() as temp_dir:
        # Given
        config_path = "oumi://smollm/inference/135m_infer.yaml"
        expected_path = Path(temp_dir) / "smollm/inference/135m_infer.yaml"
        monkeypatch.setenv("OUMI_DIR", temp_dir)

        # When
        result = runner.invoke(app, [config_path])

        # Then
        assert result.exit_code == 0
        mock_requests.get.assert_called_once()
        assert expected_path.exists()


def test_fetch_with_default_dir(app, mock_requests, monkeypatch):
    # Given
    config_path = "oumi://smollm/inference/135m_infer.yaml"
    expected_path = Path.home() / ".oumi/configs/smollm/inference/135m_infer.yaml"
    monkeypatch.delenv("OUMI_DIR", raising=False)

    # When
    result = runner.invoke(app, [config_path])

    # Then
    assert result.exit_code == 0
    mock_requests.get.assert_called_once()
    assert expected_path.exists()

    # Cleanup
    if expected_path.exists():
        expected_path.unlink()
    if expected_path.parent.exists():
        expected_path.parent.rmdir()
