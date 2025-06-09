import tempfile
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from oumi.cli.cli_utils import CONTEXT_ALLOW_EXTRA_ARGS
from oumi.cli.judge import conversations, dataset, model
from oumi.core.types import Conversation, Message
from oumi.core.types.conversation import Role
from oumi.utils.io_utils import save_jsonlines

runner = CliRunner()

config = "oumi/v1_xml_unit_test"


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


def test_judge_dataset_runs(app):
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

    assert result.exit_code == 0, f"CLI command failed with: {result.exception}"


def test_judge_dataset_with_output_file(app):
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

        assert result.exit_code == 0
        assert Path(output_file).exists()


def test_judge_conversations_runs(app):
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

        assert result.exit_code == 0


def test_judge_conversations_with_output_file(app):
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

        assert result.exit_code == 0
        assert Path(output_file).exists()
