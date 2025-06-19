import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from oumi.cli.cli_utils import CONTEXT_ALLOW_EXTRA_ARGS
from oumi.cli.judge_v2 import judge_file
from oumi.judges_v2.base_judge import JudgeOutput

runner = CliRunner()


@pytest.fixture
def mock_parse_extra_cli_args():
    with patch("oumi.cli.cli_utils.parse_extra_cli_args") as m_parse:
        m_parse.return_value = {}
        yield m_parse


@pytest.fixture
def app():
    import typer

    judge_app = typer.Typer()
    judge_app.command(context_settings=CONTEXT_ALLOW_EXTRA_ARGS)(judge_file)
    yield judge_app


@pytest.fixture
def mock_judge_file():
    with patch("oumi.judge_v2.judge_file") as m_jf:
        yield m_jf


@pytest.fixture
def mock_judge_config_from_path():
    with patch("oumi.core.configs.judge_config_v2.JudgeConfig.from_path") as m_rjc:
        yield m_rjc


@pytest.fixture
def sample_judge_output():
    return JudgeOutput(
        raw_output="Test judgment",
        parsed_output={"quality": "good"},
        field_values={"quality": "good"},
        field_scores={"quality": 0.5},
    )


def test_judge_file(
    app,
    mock_parse_extra_cli_args,
    mock_judge_file,
    mock_judge_config_from_path,
    sample_judge_output,
):
    """Test that judge_file command runs successfully with all required parameters."""
    judge_config = "judge_config.yaml"
    input_file = "input.jsonl"

    mock_judge_file.return_value = [sample_judge_output]

    with patch("oumi.cli.judge_v2.Path") as mock_path:
        mock_path.return_value.exists.return_value = True
        result = runner.invoke(
            app,
            [
                "--judge-config",
                judge_config,
                "--input-file",
                input_file,
            ],
        )

        assert result.exit_code == 0
        mock_parse_extra_cli_args.assert_called_once()
        mock_judge_config_from_path.assert_called_once_with(
            path=judge_config, extra_args={}
        )

        mock_judge_file.assert_called_once_with(
            judge_config=mock_judge_config_from_path.return_value,
            input_file=input_file,
            output_file=None,
        )


def test_judge_file_with_output_file(
    app,
    mock_parse_extra_cli_args,
    mock_judge_file,
    mock_judge_config_from_path,
    sample_judge_output,
):
    """Test that judge_file saves results to output file when specified."""
    with tempfile.TemporaryDirectory() as temp_dir:
        judge_config = "judge_config.yaml"
        input_file = "input.jsonl"
        output_file = str(Path(temp_dir) / "output.jsonl")

        mock_judge_file.return_value = [sample_judge_output]

        with patch("oumi.cli.judge_v2.Path") as mock_path:
            mock_path.return_value.exists.return_value = True
            result = runner.invoke(
                app,
                [
                    "--judge-config",
                    judge_config,
                    "--input-file",
                    input_file,
                    "--output-file",
                    output_file,
                ],
            )

            assert result.exit_code == 0
            mock_parse_extra_cli_args.assert_called_once()
            mock_judge_config_from_path.assert_called_once_with(
                path=judge_config, extra_args={}
            )

            mock_judge_file.assert_called_once_with(
                judge_config=mock_judge_config_from_path.return_value,
                input_file=input_file,
                output_file=output_file,
            )
