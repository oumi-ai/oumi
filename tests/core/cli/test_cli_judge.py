# import json
# import tempfile
# from pathlib import Path
from unittest.mock import patch

import pytest
import typer
from typer.testing import CliRunner

from oumi.core.cli.cli_utils import CONTEXT_ALLOW_EXTRA_ARGS
from oumi.core.cli.judge import conversations, dataset

runner = CliRunner()


#
# Fixtures
#
@pytest.fixture
def app():
    judge_app = typer.Typer()
    judge_app.command(context_settings=CONTEXT_ALLOW_EXTRA_ARGS)(dataset)
    judge_app.command(context_settings=CONTEXT_ALLOW_EXTRA_ARGS)(conversations)
    yield judge_app


@pytest.fixture
def mock_judge_dataset():
    with patch("oumi.core.cli.judge.dataset") as mock_dataset:
        yield mock_dataset


@pytest.fixture
def mock_judge_conversations():
    with patch("oumi.core.cli.judge.conversations") as mock_conversations:
        yield mock_conversations


def test_judge_dataset_runs(app, mock_judge_dataset):
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

    assert result.exit_code == 0
    mock_judge_dataset.assert_called()
    mock_judge_dataset.assert_called_once_with(config, dataset_name="debug_sft")
    # mock_registry.get_dataset.assert_called_once_with("test_dataset", subset=None)


# def test_judge_dataset_with_output_file(app, mock_judge_dataset, mock_registry):
#     with tempfile.TemporaryDirectory() as output_temp_dir:
#         output_file = str(Path(output_temp_dir) / "output.jsonl")

#         mock_dataset = Mock()
#         mock_registry.get_dataset.return_value.return_value = mock_dataset
#         mock_judge_dataset.return_value = [{"result": "test"}]

#         result = runner.invoke(
#             app,
#             [
#                 "dataset",
#                 "--config",
#                 config_path,
#                 "--dataset_name",
#                 "test_dataset",
#                 "--output_file",
#                 output_file,
#             ],
#         )

#         assert result.exit_code == 0
#         mock_judge_dataset.assert_called_once_with(config, dataset=mock_dataset)
#         assert Path(output_file).exists()


# def test_judge_conversations_runs(app, mock_judge_conversations):
#     with tempfile.TemporaryDirectory() as output_temp_dir:
#         config_path = str(Path(output_temp_dir) / "judge_config.yaml")
#         config = unit_test_judge()
#         config.to_yaml(config_path)

#         input_file = str(Path(output_temp_dir) / "input.jsonl")
#         with open(input_file, "w") as f:
#             json.dump([{"messages": [{"role": "user", "content": "Hello"}]}], f)

#         result = runner.invoke(
#             app,
#             [
#                 "conversations",
#                 "--config",
#                 config_path,
#                 "--input_file",
#                 input_file,
#             ],
#         )

#         assert result.exit_code == 0
#         mock_judge_conversations.assert_called_once()


# def test_judge_conversations_with_output_file(app, mock_judge_conversations):
#     with tempfile.TemporaryDirectory() as output_temp_dir:
#         config_path = str(Path(output_temp_dir) / "judge_config.yaml")
#         config = unit_test_judge()
#         config.to_yaml(config_path)

#         input_file = str(Path(output_temp_dir) / "input.jsonl")
#         with open(input_file, "w") as f:
#             json.dump([{"messages": [{"role": "user", "content": "Hello"}]}], f)

#         output_file = str(Path(output_temp_dir) / "output.jsonl")

#         mock_judge_conversations.return_value = [{"result": "test"}]

#         result = runner.invoke(
#             app,
#             [
#                 "conversations",
#                 "--config",
#                 config_path,
#                 "--input_file",
#                 input_file,
#                 "--output_file",
#                 output_file,
#             ],
#         )

#         assert result.exit_code == 0
#         mock_judge_conversations.assert_called_once()
#         assert Path(output_file).exists()


# def test_judge_dataset_missing_dataset_name(app):
#     with tempfile.TemporaryDirectory() as output_temp_dir:
#         config_path = str(Path(output_temp_dir) / "judge_config.yaml")
#         config = unit_test_judge()
#         config.to_yaml(config_path)

#         result = runner.invoke(
#             app,
#             [
#                 "dataset",
#                 "--config",
#                 config_path,
#             ],
#         )

#         assert result.exit_code != 0
#         assert "Dataset name is required" in result.output


# def test_judge_conversations_missing_input_file(app, unit_test_judge):
#     with tempfile.TemporaryDirectory() as output_temp_dir:
#         config_path = str(Path(output_temp_dir) / "judge_config.yaml")
#         unit_test_judge.to_yaml(config_path)

#         result = runner.invoke(
#             app,
#             [
#                 "conversations",
#                 "--config",
#                 config_path,
#             ],
#         )

#         assert result.exit_code != 0
#         assert "Input file is required" in result.output


# def test_judge_invalid_config(app):
#     result = runner.invoke(
#         app,
#         [
#             "dataset",
#             "--config",
#             "invalid_config",
#             "--dataset_name",
#             "test_dataset",
#         ],
#     )

#     assert result.exit_code != 0
#     assert "Config file not found" in result.output


# def test_judge_dataset_with_extra_args(app, mock_judge_dataset, mock_registry):
#     with tempfile.TemporaryDirectory() as output_temp_dir:
#         config_path = str(Path(output_temp_dir) / "judge_config.yaml")
#         config = unit_test_judge()
#         config.to_yaml(config_path)

#         mock_dataset = Mock()
#         mock_registry.get_dataset.return_value.return_value = mock_dataset

#         result = runner.invoke(
#             app,
#             [
#                 "dataset",
#                 "--config",
#                 config_path,
#                 "--dataset_name",
#                 "test_dataset",
#                 "--model.model_name",
#                 "new_model_name",
#             ],
#         )

#         assert result.exit_code == 0
#         called_config = mock_judge_dataset.call_args[0][0]
#         assert called_config.model.model_name == "new_model_name"


# def test_judge_conversations_with_extra_args(app, mock_judge_conversations):
#     with tempfile.TemporaryDirectory() as output_temp_dir:
#         config_path = str(Path(output_temp_dir) / "judge_config.yaml")
#         config = unit_test_judge()
#         config.to_yaml(config_path)

#         input_file = str(Path(output_temp_dir) / "input.jsonl")
#         with open(input_file, "w") as f:
#             json.dump([{"messages": [{"role": "user", "content": "Hello"}]}], f)

#         result = runner.invoke(
#             app,
#             [
#                 "conversations",
#                 "--config",
#                 config_path,
#                 "--input_file",
#                 input_file,
#                 "--model.model_name",
#                 "new_model_name",
#             ],
#         )

#         assert result.exit_code == 0
#         called_config = mock_judge_conversations.call_args[0][0]
#         assert called_config.model.model_name == "new_model_name"
