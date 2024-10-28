from inspect import signature
from typing import Callable
from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

from oumi.core.cli.evaluate import evaluate
from oumi.core.cli.infer import infer
from oumi.core.cli.judge import conversations, dataset
from oumi.core.cli.launch import cancel, down, status, up, which
from oumi.core.cli.launch import run as launcher_run
from oumi.core.cli.main import get_app
from oumi.core.cli.train import train

runner = CliRunner()


def _copy_command(mock: Mock, command: Callable):
    mock.__name__ = command.__name__
    mock.__annotations__ = command.__annotations__
    mock.__signature__ = signature(command)


#
# Fixtures
#
@pytest.fixture
def mock_train():
    with patch("oumi.core.cli.main.train") as m_train:
        _copy_command(m_train, train)
        yield m_train


@pytest.fixture
def mock_eval():
    with patch("oumi.core.cli.main.evaluate") as m_eval:
        _copy_command(m_eval, evaluate)
        yield m_eval


@pytest.fixture
def mock_infer():
    with patch("oumi.core.cli.main.infer") as m_infer:
        _copy_command(m_infer, infer)
        yield m_infer


@pytest.fixture
def mock_down():
    with patch("oumi.core.cli.main.down") as m_down:
        _copy_command(m_down, down)
        yield m_down


@pytest.fixture
def mock_launcher_run():
    with patch("oumi.core.cli.main.launcher_run") as m_launcher_run:
        _copy_command(m_launcher_run, launcher_run)
        yield m_launcher_run


@pytest.fixture
def mock_status():
    with patch("oumi.core.cli.main.status") as m_status:
        _copy_command(m_status, status)
        yield m_status


@pytest.fixture
def mock_cancel():
    with patch("oumi.core.cli.main.cancel") as m_cancel:
        _copy_command(m_cancel, cancel)
        yield m_cancel


@pytest.fixture
def mock_up():
    with patch("oumi.core.cli.main.up") as m_up:
        _copy_command(m_up, up)
        yield m_up


@pytest.fixture
def mock_which():
    with patch("oumi.core.cli.main.which") as m_which:
        _copy_command(m_which, which)
        yield m_which


@pytest.fixture
def mock_judge_dataset():
    with patch("oumi.core.cli.main.dataset") as m_dataset:
        _copy_command(m_dataset, dataset)
        yield m_dataset


@pytest.fixture
def mock_judge_conversations():
    with patch("oumi.core.cli.main.conversations") as m_conversations:
        _copy_command(m_conversations, conversations)
        yield m_conversations


def test_main_train_registered(mock_train):
    _ = runner.invoke(
        get_app(), ["train", "--config", "some/path", "--allow_extra" "args"]
    )
    mock_train.assert_called_once()


def test_main_infer_registered(mock_infer):
    _ = runner.invoke(
        get_app(), ["infer", "--config", "some/path", "--allow_extra" "args"]
    )
    mock_infer.assert_called_once()


def test_main_eval_registered(mock_eval):
    _ = runner.invoke(
        get_app(), ["eval", "--config", "some/path", "--allow_extra" "args"]
    )
    mock_eval.assert_called_once()


def test_main_evaluate_registered(mock_eval):
    _ = runner.invoke(
        get_app(), ["evaluate", "--config", "some/path", "--allow_extra" "args"]
    )
    mock_eval.assert_called_once()


def test_main_launch_registered():
    result = runner.invoke(get_app(), ["launch", "--help"])
    for cmd in ["down", "run", "status", "cancel", "up", "which"]:
        assert cmd in result.output


def test_main_down_registered(mock_down):
    _ = runner.invoke(
        get_app(), ["launch", "down", "--cluster", "cluster", "--cloud", "gcp"]
    )
    mock_down.assert_called_once()


def test_main_run_registered(mock_launcher_run):
    _ = runner.invoke(
        get_app(),
        ["launch", "run", "--config", "some_path", "--cluster", "clust", "--detach"],
    )
    mock_launcher_run.assert_called_once()


def test_main_status_registered(mock_status):
    _ = runner.invoke(
        get_app(),
        [
            "launch",
            "status",
            "--cloud",
            "gcp",
            "--cluster",
            "cluster",
            "--id",
            "foobar",
        ],
    )
    mock_status.assert_called_once()


def test_main_cancel_registered(mock_cancel):
    _ = runner.invoke(
        get_app(),
        [
            "launch",
            "cancel",
            "--cloud",
            "gcp",
            "--cluster",
            "cluster",
            "--id",
            "foobar",
        ],
    )
    mock_cancel.assert_called_once()


def test_main_up_registered(mock_up):
    _ = runner.invoke(
        get_app(),
        ["launch", "up", "--config", "some_path", "--cluster", "clust", "--detach"],
    )
    mock_up.assert_called_once()


def test_main_which_registered(mock_which):
    _ = runner.invoke(get_app(), ["launch", "which"])
    mock_which.assert_called_once()


def test_main_judge_dataset_registered(mock_judge_dataset):
    _ = runner.invoke(
        get_app(),
        [
            "judge",
            "dataset",
            "--config",
            "some_config",
            "--dataset-name",
            "some_dataset",
        ],
    )
    mock_judge_dataset.assert_called_once()


def test_main_judge_conversations_registered(mock_judge_conversations):
    _ = runner.invoke(
        get_app(),
        [
            "judge",
            "conversations",
            "--config",
            "some_config",
            "--input-file",
            "some_file.jsonl",
        ],
    )
    mock_judge_conversations.assert_called_once()
