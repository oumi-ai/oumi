import tempfile
from pathlib import Path
from unittest.mock import call, patch

import pytest
from typer.testing import CliRunner

from oumi.cli.cli_utils import CONTEXT_ALLOW_EXTRA_ARGS
from oumi.cli.judge_v2 import judge_file
from oumi.judges_v2.base_judge import JudgeOutput

runner = CliRunner()


@pytest.fixture
def mock_fetch_config():
    with patch("oumi.cli.cli_utils.resolve_and_fetch_config") as m_fetch:
        m_fetch.side_effect = lambda x: x
        yield m_fetch


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
def mock_load_configs():
    with (
        patch("oumi.cli.judge_v2._resolve_judge_config") as m_rjc,
        patch("oumi.cli.judge_v2._load_inference_config") as m_lic,
    ):
        yield m_rjc, m_lic


@pytest.fixture
def sample_judge_output():
    return JudgeOutput(
        raw_output="Test judgment",
        parsed_output={"quality": "good"},
        field_values={"quality": "good"},
        field_scores={"quality": 0.5},
    )


def test_judge_file(
    app, mock_fetch_config, mock_judge_file, mock_load_configs, sample_judge_output
):
    """Test that judge_file command runs successfully with all required parameters."""
    judge_config = "judge_config.yaml"
    inference_config = "inference_config.yaml"
    input_file = "input.jsonl"

    mock_judge_file.return_value = [sample_judge_output]
    mock_resolve_judge_config, mock_load_inference_config = mock_load_configs

    with patch("oumi.cli.judge_v2.Path") as mock_path:
        mock_path.return_value.exists.return_value = True
        result = runner.invoke(
            app,
            [
                "--judge-config",
                judge_config,
                "--inference-config",
                inference_config,
                "--input-file",
                input_file,
            ],
        )

        assert result.exit_code == 0
        mock_fetch_config.assert_has_calls([call(inference_config)])
        mock_resolve_judge_config.assert_called_once()
        mock_load_inference_config.assert_called_once()

        mock_judge_file.assert_called_once_with(
            judge_config=mock_resolve_judge_config.return_value,
            inference_config=mock_load_inference_config.return_value,
            input_file=input_file,
            output_file=None,
        )


def test_judge_file_with_output_file(
    app, mock_fetch_config, mock_judge_file, mock_load_configs, sample_judge_output
):
    """Test that judge_file saves results to output file when specified."""
    with tempfile.TemporaryDirectory() as temp_dir:
        judge_config = "judge_config.yaml"
        inference_config = "inference_config.yaml"
        input_file = "input.jsonl"
        output_file = str(Path(temp_dir) / "output.jsonl")

        mock_judge_file.return_value = [sample_judge_output]
        mock_resolve_judge_config, mock_load_inference_config = mock_load_configs

        with patch("oumi.cli.judge_v2.Path") as mock_path:
            mock_path.return_value.exists.return_value = True
            result = runner.invoke(
                app,
                [
                    "--judge-config",
                    judge_config,
                    "--inference-config",
                    inference_config,
                    "--input-file",
                    input_file,
                    "--output-file",
                    output_file,
                ],
            )

            assert result.exit_code == 0
            mock_fetch_config.assert_has_calls([call(inference_config)])
            mock_resolve_judge_config.assert_called_once()
            mock_load_inference_config.assert_called_once()

            mock_judge_file.assert_called_once_with(
                judge_config=mock_resolve_judge_config.return_value,
                inference_config=mock_load_inference_config.return_value,
                input_file=input_file,
                output_file=output_file,
            )
