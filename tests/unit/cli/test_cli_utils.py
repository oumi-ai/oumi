import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, call, patch

import pytest
import typer
import yaml
from requests.exceptions import RequestException
from typer.testing import CliRunner

from oumi.cli.cli_utils import (
    CONFIG_FLAGS,
    CONTEXT_ALLOW_EXTRA_ARGS,
    LogLevel,
    configure_common_env_vars,
    parse_extra_cli_args,
    resolve_and_fetch_config,
    section_header,
)


@pytest.fixture
def mock_response():
    response = Mock()
    response.text = "key: value"
    response.raise_for_status.return_value = None
    return response


@pytest.fixture
def mock_requests(mock_response):
    with patch("oumi.cli.cli_utils.requests") as r_mock:
        r_mock.get.return_value = mock_response
        yield r_mock


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


@patch.dict(os.environ, {"FOO": "1"}, clear=True)
def test_configure_common_env_vars_empty():
    configure_common_env_vars()
    assert os.environ == {
        "FOO": "1",
        "ACCELERATE_LOG_LEVEL": "info",
        "TOKENIZERS_PARALLELISM": "false",
    }


@patch.dict(
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


@patch.dict(
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


def test_resolve_and_fetch_config_with_oumi_prefix_and_explicit_output_dir(
    mock_requests,
):
    with tempfile.TemporaryDirectory() as temp_dir:
        # Given
        output_dir = Path(temp_dir)
        config_path = "oumi://configs/recipes/smollm/inference/135m_infer.yaml"
        expected_path = output_dir / "configs/recipes/smollm/inference/135m_infer.yaml"

        # When
        result = resolve_and_fetch_config(config_path, output_dir)

        # Then
        assert result == expected_path
        mock_requests.get.assert_called_once()
        assert expected_path.exists()


def test_resolve_and_fetch_config_without_prefix_and_explicit_output_dir(mock_requests):
    with tempfile.TemporaryDirectory() as temp_dir:
        # Given
        output_dir = Path(temp_dir)
        config_path = (
            "configs/recipes/smollm/inference/135m_infer.yaml"  # No oumi:// prefix
        )

        # When
        result = resolve_and_fetch_config(config_path, output_dir)

        # Then
        assert result == Path(config_path)
        assert not mock_requests.get.called


def test_resolve_and_fetch_config_with_oumi_prefix_and_env_dir(
    mock_requests, monkeypatch
):
    with tempfile.TemporaryDirectory() as temp_dir:
        # Given
        config_path = "oumi://configs/recipes/smollm/inference/135m_infer.yaml"
        expected_path = (
            Path(temp_dir) / "configs/recipes/smollm/inference/135m_infer.yaml"
        )
        monkeypatch.setenv("OUMI_DIR", temp_dir)

        # When
        result = resolve_and_fetch_config(config_path)

        # Then
        assert result == expected_path
        mock_requests.get.assert_called_once()
        assert expected_path.exists()


def test_resolve_and_fetch_config_without_prefix_and_env_dir(
    mock_requests, monkeypatch
):
    with tempfile.TemporaryDirectory() as temp_dir:
        # Given
        config_path = (
            "configs/recipes/smollm/inference/135m_infer.yaml"  # No oumi:// prefix
        )
        monkeypatch.setenv("OUMI_DIR", temp_dir)

        # When
        result = resolve_and_fetch_config(config_path)

        # Then
        assert result == Path(config_path)
        assert not mock_requests.get.called


def test_resolve_and_fetch_config_with_oumi_prefix_and_default_dir(
    mock_requests, monkeypatch
):
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch("oumi.cli.cli_utils.OUMI_FETCH_DIR", temp_dir):
            # Given
            config_path = "oumi://configs/recipes/smollm/inference/135m_infer.yaml"
            expected_path = (
                Path(temp_dir) / "configs/recipes/smollm/inference/135m_infer.yaml"
            )
            monkeypatch.delenv("OUMI_DIR", raising=False)

            # When
            result = resolve_and_fetch_config(config_path)

            # Then
            assert result == expected_path
            mock_requests.get.assert_called_once()
            assert expected_path.exists()


def test_resolve_and_fetch_config_without_prefix_and_default_dir(
    mock_requests, monkeypatch
):
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch("oumi.cli.cli_utils.OUMI_FETCH_DIR", temp_dir):
            # Given
            config_path = (
                "configs/recipes/smollm/inference/135m_infer.yaml"  # No oumi:// prefix
            )
            expected_path = Path(config_path)
            monkeypatch.delenv("OUMI_DIR", raising=False)

            # When
            result = resolve_and_fetch_config(config_path)

            # Then
            assert result == expected_path
            assert not mock_requests.get.called


def test_resolve_and_fetch_config_with_existing_file_default_force(mock_requests):
    with tempfile.TemporaryDirectory() as temp_dir:
        # Given
        output_dir = Path(temp_dir)
        config_path = "oumi://configs/recipes/smollm/inference/135m_infer.yaml"
        expected_path = output_dir / "configs/recipes/smollm/inference/135m_infer.yaml"

        # Create existing file
        expected_path.parent.mkdir(parents=True)
        expected_path.write_text("existing content")

        # When
        result = resolve_and_fetch_config(config_path, output_dir)
        # Then
        assert result == expected_path
        assert mock_requests.get.call_count == 1
        assert expected_path.read_text() == "key: value"


def test_resolve_and_fetch_config_with_existing_file_force(mock_requests):
    with tempfile.TemporaryDirectory() as temp_dir:
        # Given
        output_dir = Path(temp_dir)
        config_path = "oumi://configs/recipes/smollm/inference/135m_infer.yaml"
        expected_path = output_dir / "configs/recipes/smollm/inference/135m_infer.yaml"

        # Create existing file
        expected_path.parent.mkdir(parents=True)
        expected_path.write_text("existing content")

        # When
        result = resolve_and_fetch_config(config_path, output_dir, force=True)

        # Then
        assert result == expected_path
        mock_requests.get.assert_called_once()
        assert expected_path.exists()
        assert expected_path.read_text() == "key: value"  # From mock_response


def test_resolve_and_fetch_config_force_no_conflict(mock_requests):
    with tempfile.TemporaryDirectory() as temp_dir:
        # Given
        output_dir = Path(temp_dir)
        config_path = "oumi://configs/recipes/smollm/inference/135m_infer.yaml"
        expected_path = output_dir / "configs/recipes/smollm/inference/135m_infer.yaml"

        # When
        result = resolve_and_fetch_config(config_path, output_dir, force=True)
        # Then
        assert result == expected_path
        assert mock_requests.get.call_count == 1
        assert expected_path.read_text() == "key: value"


def test_resolve_and_fetch_config_conflict_no_force(mock_requests):
    with pytest.raises(RuntimeError, match="Config already exists at"):
        with tempfile.TemporaryDirectory() as temp_dir:
            # Given
            output_dir = Path(temp_dir)
            config_path = "oumi://configs/recipes/smollm/inference/135m_infer.yaml"
            expected_path = (
                output_dir / "configs/recipes/smollm/inference/135m_infer.yaml"
            )

            # Create existing file
            expected_path.parent.mkdir(parents=True)
            expected_path.write_text("existing content")

            # When
            result = resolve_and_fetch_config(config_path, output_dir, force=False)
            # Then
            assert result == expected_path
            assert mock_requests.get.call_count == 1
            assert expected_path.read_text() == "key: value"


def test_resolve_and_fetch_config_http_error(mock_requests):
    with tempfile.TemporaryDirectory() as temp_dir:
        # Given
        output_dir = Path(temp_dir)
        config_path = "oumi://configs/recipes/smollm/inference/135m_infer.yaml"
        mock_requests.get.side_effect = RequestException("HTTP Error")

        # When
        with pytest.raises(RequestException):
            _ = resolve_and_fetch_config(config_path, output_dir, force=False)


def test_resolve_and_fetch_config_yaml_error(mock_requests):
    with tempfile.TemporaryDirectory() as temp_dir:
        # Given
        output_dir = Path(temp_dir)
        config_path = "oumi://configs/recipes/smollm/inference/135m_infer.yaml"
        mock_requests.get.return_value.text = "foo: bar\nbye"
        # When
        with pytest.raises(yaml.YAMLError):
            _ = resolve_and_fetch_config(config_path, output_dir, force=False)


def test_section_header():
    """Test the section_header function."""

    # Create a mock console
    mock_console = Mock()
    mock_console.width = 10

    # Call the function with a test message
    test_message = "Test Header"
    section_header(test_message, console=mock_console)

    # Assert that the console's print method was called with the correct message
    mock_console.print.assert_has_calls(
        [
            call("\n[blue]━━━━━━━━━━[/blue]"),
            call("[yellow]   Test Header[/yellow]"),
            call("[blue]━━━━━━━━━━[/blue]\n"),
        ]
    )
