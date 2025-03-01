import logging
import tempfile
from pathlib import Path
from unittest.mock import call, patch

import pytest
import typer
from typer.testing import CliRunner

import oumi.core.registry
import oumi.judge
from oumi.cli.cli_utils import CONTEXT_ALLOW_EXTRA_ARGS
from oumi.cli.judge import conversations, dataset, model
from oumi.core.types import Conversation, Message
from oumi.core.types.conversation import Role
from oumi.utils.io_utils import save_jsonlines
from oumi.utils.logging import logger

runner = CliRunner()

config = "oumi/v1_xml_unit_test"


@pytest.fixture
def mock_fetch():
    with patch("oumi.cli.cli_utils.resolve_and_fetch_config") as m_fetch:
        m_fetch.side_effect = lambda x: x
        yield m_fetch


#
# Fixtures
#
@pytest.fixture
def app():
    judge_app = typer.Typer()
    judge_app.command(context_settings=CONTEXT_ALLOW_EXTRA_ARGS)(dataset)
    judge_app.command(context_settings=CONTEXT_ALLOW_EXTRA_ARGS)(conversations)
    judge_app.command(context_settings=CONTEXT_ALLOW_EXTRA_ARGS)(model)
    yield judge_app


@pytest.fixture
def mock_registry():
    with patch.object(oumi.core.registry, "REGISTRY") as m_registry:
        yield m_registry


@pytest.fixture
def mock_judge_dataset():
    with patch.object(oumi.judge, "judge_dataset", autospec=True) as m_jd:
        yield m_jd


@pytest.fixture
def mock_judge_conversations():
    with patch.object(oumi.judge, "judge_conversations", autospec=True) as m_jc:
        yield m_jc


def test_judge_model_calls_fetch(app, mock_registry, mock_fetch):
    returned_exception = RuntimeError("fetch failed")
    mock_fetch.side_effect = ["oumi/v1_xml_unit_test", returned_exception]
    config = "oumi/v1_xml_unit_test"
    mock_fetch.return_value = config
    result = runner.invoke(
        app,
        [
            "model",
            "--config",
            config,
            "--inference-config",
            config,
        ],
    )
    assert result.exit_code == 1
    assert returned_exception == result.exception
    mock_fetch.assert_has_calls([call(config), call(config)])


def test_judge_dataset_runs(app, mock_registry, mock_judge_dataset, mock_fetch):
    config = "oumi/v1_xml_unit_test"
    result = runner.invoke(
        app,
        [
            "dataset",
            "--config",
            config,
            "--dataset-name",
            "debug_sft",
        ],
    )
    mock_judge_dataset.assert_called_once()
    mock_fetch.assert_called_once_with(config)
    assert result.exit_code == 0, f"CLI command failed with: {result.exception}"


def test_judge_logging_levels(
    app, mock_registry, mock_judge_dataset, mock_judge_conversations
):
    config = "oumi/v1_xml_unit_test"
    _ = runner.invoke(
        app,
        [
            "dataset",
            "--config",
            config,
            "--dataset-name",
            "debug_sft",
            "--log-level",
            "DEBUG",
        ],
    )
    assert logger.level == logging.DEBUG

    with tempfile.TemporaryDirectory() as output_temp_dir:
        input_file = str(Path(output_temp_dir) / "input.jsonl")

        conversation = Conversation(
            messages=[
                Message(role=Role.USER, content="Hello"),
                Message(role=Role.ASSISTANT, content="Hello"),
            ]
        )

        save_jsonlines(
            input_file,
            [conversation.to_dict()],
        )

        _ = runner.invoke(
            app,
            [
                "conversations",
                "--config",
                config,
                "--input-file",
                input_file,
                "-log",
                "INFO",
            ],
        )
        assert logger.level == logging.INFO


def test_judge_dataset_with_output_file(app, mock_registry, mock_judge_dataset):
    with tempfile.TemporaryDirectory() as output_temp_dir:
        output_file = str(Path(output_temp_dir) / "output.jsonl")

        result = runner.invoke(
            app,
            [
                "dataset",
                "--config",
                config,
                "--dataset-name",
                "debug_sft",
                "--output-file",
                output_file,
            ],
        )
        mock_judge_dataset.assert_called_once()
        assert result.exit_code == 0
        assert Path(output_file).exists()


def test_judge_conversations_runs(app, mock_judge_conversations, mock_fetch):
    with tempfile.TemporaryDirectory() as output_temp_dir:
        input_file = str(Path(output_temp_dir) / "input.jsonl")

        conversation = Conversation(
            messages=[
                Message(role=Role.USER, content="Hello"),
                Message(role=Role.ASSISTANT, content="Hello"),
            ]
        )

        save_jsonlines(
            input_file,
            [conversation.to_dict()],
        )
        result = runner.invoke(
            app,
            [
                "conversations",
                "--config",
                config,
                "--input-file",
                input_file,
            ],
        )
        mock_judge_conversations.assert_called_once()
        mock_fetch.assert_called_once_with(config)
        assert result.exit_code == 0


def test_judge_conversations_with_output_file(app, mock_judge_conversations):
    with tempfile.TemporaryDirectory() as output_temp_dir:
        input_file = str(Path(output_temp_dir) / "input.jsonl")
        conversation = Conversation(
            messages=[
                Message(role=Role.USER, content="Hello"),
                Message(role=Role.ASSISTANT, content="Hello"),
            ]
        )

        save_jsonlines(
            input_file,
            [conversation.to_dict()],
        )

        output_file = str(Path(output_temp_dir) / "output.jsonl")

        result = runner.invoke(
            app,
            [
                "conversations",
                "--config",
                config,
                "--input-file",
                input_file,
                "--output-file",
                output_file,
            ],
        )
        mock_judge_conversations.assert_called_once()
        assert result.exit_code == 0
        assert Path(output_file).exists()


def test_judge_dataset_missing_dataset_name(app):
    result = runner.invoke(
        app,
        [
            "dataset",
            "--config",
            config,
        ],
    )

    assert result.exit_code != 0
    assert "Dataset name is required" in result.output


def test_judge_conversations_missing_input_file(app):
    result = runner.invoke(
        app,
        [
            "conversations",
            "--config",
            config,
        ],
    )

    assert result.exit_code != 0
    assert "Input file is required" in result.output


def test_judge_invalid_config(app):
    result = runner.invoke(
        app,
        [
            "dataset",
            "--config",
            "invalid_config",
            "--dataset-name",
            "test_dataset",
        ],
    )

    assert result.exit_code != 0
    assert "Config file not found" in result.output
