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


def test_fetch_with_oumi_prefix_and_explicit_output_dir(app, mock_requests):
    with tempfile.TemporaryDirectory() as temp_dir:
        # Given
        output_dir = Path(temp_dir)
        config_path = "oumi://configs/recipes/smollm/inference/135m_infer.yaml"
        expected_path = output_dir / "configs/recipes/smollm/inference/135m_infer.yaml"

        # When
        result = runner.invoke(app, [config_path, "-o", str(output_dir)])

        # Then
        assert result.exit_code == 0
        mock_requests.get.assert_called_once()
        assert expected_path.exists()


def test_fetch_without_prefix_and_explicit_output_dir(app, mock_requests):
    with tempfile.TemporaryDirectory() as temp_dir:
        # Given
        output_dir = Path(temp_dir)
        config_path = (
            "configs/recipes/smollm/inference/135m_infer.yaml"  # No oumi:// prefix
        )
        expected_path = output_dir / "configs/recipes/smollm/inference/135m_infer.yaml"

        # When
        result = runner.invoke(app, [config_path, "-o", str(output_dir)])

        # Then
        assert result.exit_code == 0
        mock_requests.get.assert_called_once()
        assert expected_path.exists()


def test_fetch_with_oumi_prefix_and_env_dir(app, mock_requests, monkeypatch):
    with tempfile.TemporaryDirectory() as temp_dir:
        # Given
        config_path = "oumi://configs/recipes/smollm/inference/135m_infer.yaml"
        expected_path = (
            Path(temp_dir) / "configs/recipes/smollm/inference/135m_infer.yaml"
        )
        monkeypatch.setenv("OUMI_DIR", temp_dir)

        # When
        result = runner.invoke(app, [config_path])

        # Then
        assert result.exit_code == 0
        mock_requests.get.assert_called_once()
        assert expected_path.exists()


def test_fetch_without_prefix_and_env_dir(app, mock_requests, monkeypatch):
    with tempfile.TemporaryDirectory() as temp_dir:
        # Given
        config_path = (
            "configs/recipes/smollm/inference/135m_infer.yaml"  # No oumi:// prefix
        )
        expected_path = (
            Path(temp_dir) / "configs/recipes/smollm/inference/135m_infer.yaml"
        )
        monkeypatch.setenv("OUMI_DIR", temp_dir)

        # When
        result = runner.invoke(app, [config_path])

        # Then
        assert result.exit_code == 0
        mock_requests.get.assert_called_once()
        assert expected_path.exists()


def test_fetch_with_oumi_prefix_and_default_dir(app, mock_requests, monkeypatch):
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch("oumi.cli.cli_utils.OUMI_FETCH_DIR", temp_dir):
            # Given
            config_path = "oumi://configs/recipes/smollm/inference/135m_infer.yaml"
            expected_path = (
                Path(temp_dir) / "configs/recipes/smollm/inference/135m_infer.yaml"
            )
            monkeypatch.delenv("OUMI_DIR", raising=False)

            # When
            result = runner.invoke(app, [config_path])

            # Then
            assert result.exit_code == 0
            mock_requests.get.assert_called_once()
            assert expected_path.exists()


def test_fetch_without_prefix_and_default_dir(app, mock_requests, monkeypatch):
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch("oumi.cli.cli_utils.OUMI_FETCH_DIR", temp_dir):
            # Given
            config_path = (
                "configs/recipes/smollm/inference/135m_infer.yaml"  # No oumi:// prefix
            )
            expected_path = (
                Path(temp_dir) / "configs/recipes/smollm/inference/135m_infer.yaml"
            )
            monkeypatch.delenv("OUMI_DIR", raising=False)

            # When
            result = runner.invoke(app, [config_path])

            # Then
            assert result.exit_code == 0
            mock_requests.get.assert_called_once()
            assert expected_path.exists()


def test_fetch_with_existing_file_no_force(app, mock_requests):
    with tempfile.TemporaryDirectory() as temp_dir:
        # Given
        output_dir = Path(temp_dir)
        config_path = "oumi://configs/recipes/smollm/inference/135m_infer.yaml"
        expected_path = output_dir / "configs/recipes/smollm/inference/135m_infer.yaml"

        # Create existing file
        expected_path.parent.mkdir(parents=True)
        expected_path.write_text("existing content")

        # When
        result = runner.invoke(
            app, [config_path, "-o", str(output_dir)], catch_exceptions=False
        )
        print(result)
        # Then
        assert result.exit_code == 1
        assert "Use --force to overwrite" in result.output
        assert mock_requests.get.call_count == 0
        assert expected_path.read_text() == "existing content"


def test_fetch_with_existing_file_force(app, mock_requests):
    with tempfile.TemporaryDirectory() as temp_dir:
        # Given
        output_dir = Path(temp_dir)
        config_path = "oumi://configs/recipes/smollm/inference/135m_infer.yaml"
        expected_path = output_dir / "configs/recipes/smollm/inference/135m_infer.yaml"

        # Create existing file
        expected_path.parent.mkdir(parents=True)
        expected_path.write_text("existing content")

        # When
        result = runner.invoke(app, [config_path, "-o", str(output_dir), "--force"])

        # Then
        assert result.exit_code == 0
        mock_requests.get.assert_called_once()
        assert expected_path.exists()
        assert expected_path.read_text() == "key: value"  # From mock_response
