from inspect import signature
from typing import Callable, get_type_hints
from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

from oumi.cli.distributed_run import accelerate, torchrun
from oumi.cli.env import env
from oumi.cli.evaluate import evaluate
from oumi.cli.fetch import fetch
from oumi.cli.infer import infer
from oumi.cli.judge import conversations, dataset
from oumi.cli.launch import cancel, down, status, stop, up, which
from oumi.cli.launch import run as launcher_run
from oumi.cli.main import get_app
from oumi.cli.train import train

runner = CliRunner()


def _copy_command(mock: Mock, command: Callable):
    mock.__name__ = command.__name__
    mock.__annotations__ = get_type_hints(command)
    mock.__signature__ = signature(command)
    mock.__bool__ = lambda _: True


#
# Fixtures
#
@pytest.fixture
def mock_train():
    with patch("oumi.cli.main.train") as m_train:
        _copy_command(m_train, train)
        yield m_train


@pytest.fixture
def mock_eval():
    with patch("oumi.cli.main.evaluate") as m_eval:
        _copy_command(m_eval, evaluate)
        yield m_eval


@pytest.fixture
def mock_infer():
    with patch("oumi.cli.main.infer") as m_infer:
        _copy_command(m_infer, infer)
        yield m_infer


@pytest.fixture
def mock_fetch():
    with patch("oumi.cli.main.fetch") as m_fetch:
        _copy_command(m_fetch, fetch)
        yield m_fetch


@pytest.fixture
def mock_down():
    with patch("oumi.cli.main.down") as m_down:
        _copy_command(m_down, down)
        yield m_down


@pytest.fixture
def mock_stop():
    with patch("oumi.cli.main.stop") as m_stop:
        _copy_command(m_stop, stop)
        yield m_stop


@pytest.fixture
def mock_launcher_run():
    with patch("oumi.cli.main.launcher_run") as m_launcher_run:
        _copy_command(m_launcher_run, launcher_run)
        yield m_launcher_run


@pytest.fixture
def mock_status():
    with patch("oumi.cli.main.status") as m_status:
        _copy_command(m_status, status)
        yield m_status


@pytest.fixture
def mock_cancel():
    with patch("oumi.cli.main.cancel") as m_cancel:
        _copy_command(m_cancel, cancel)
        yield m_cancel


@pytest.fixture
def mock_up():
    with patch("oumi.cli.main.up") as m_up:
        _copy_command(m_up, up)
        yield m_up


@pytest.fixture
def mock_which():
    with patch("oumi.cli.main.which") as m_which:
        _copy_command(m_which, which)
        yield m_which


@pytest.fixture
def mock_judge_dataset():
    with patch("oumi.cli.main.dataset") as m_dataset:
        _copy_command(m_dataset, dataset)
        yield m_dataset


@pytest.fixture
def mock_judge_conversations():
    with patch("oumi.cli.main.conversations") as m_conversations:
        _copy_command(m_conversations, conversations)
        yield m_conversations


@pytest.fixture
def mock_distributed_torchrun():
    with patch("oumi.cli.main.torchrun") as m_torchrun:
        _copy_command(m_torchrun, torchrun)
        yield m_torchrun


@pytest.fixture
def mock_distributed_accelerate():
    with patch("oumi.cli.main.accelerate") as m_accelerate:
        _copy_command(m_accelerate, accelerate)
        yield m_accelerate


@pytest.fixture
def mock_env():
    with patch("oumi.cli.main.env") as m_env:
        _copy_command(m_env, env)
        yield m_env


def test_main_train_registered(mock_train):
    _ = runner.invoke(
        get_app(), ["train", "--config", "some/path", "--allow_extraargs"]
    )
    mock_train.assert_called_once()


def test_main_infer_registered(mock_infer):
    _ = runner.invoke(
        get_app(), ["infer", "--config", "some/path", "--allow_extraargs"]
    )
    mock_infer.assert_called_once()


def test_main_fetch_registered(mock_fetch):
    _ = runner.invoke(get_app(), ["fetch", "some/path", "--output-dir", "output/path"])
    mock_fetch.assert_called_once()


def test_main_eval_registered(mock_eval):
    _ = runner.invoke(get_app(), ["eval", "--config", "some/path", "--allow_extraargs"])
    mock_eval.assert_called_once()


def test_main_evaluate_registered(mock_eval):
    _ = runner.invoke(
        get_app(), ["evaluate", "--config", "some/path", "--allow_extraargs"]
    )
    mock_eval.assert_called_once()


def test_main_launch_registered():
    result = runner.invoke(get_app(), ["launch", "--help"])
    for cmd in ["down", "stop", "run", "status", "cancel", "up", "which"]:
        assert cmd in result.output


def test_main_down_registered(mock_down):
    _ = runner.invoke(
        get_app(), ["launch", "down", "--cluster", "cluster", "--cloud", "gcp"]
    )
    mock_down.assert_called_once()


def test_main_stop_registered(mock_stop):
    _ = runner.invoke(
        get_app(), ["launch", "stop", "--cluster", "cluster", "--cloud", "gcp"]
    )
    mock_stop.assert_called_once()


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


def test_main_env_registered(mock_env):
    _ = runner.invoke(get_app(), ["env"])
    mock_env.assert_called_once()


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


def test_main_distributed_registered():
    result = runner.invoke(get_app(), ["distributed", "--help"])
    for cmd in ["accelerate", "torchrun"]:
        assert cmd in result.output


def test_main_distributed_torchrun_registered(mock_distributed_torchrun):
    _ = runner.invoke(
        get_app(),
        [
            "distributed",
            "torchrun",
            "-m",
            "oumi",
            "train",
            "--config",
            "some_config",
        ],
    )
    mock_distributed_torchrun.assert_called_once()


def test_main_distributed_accelerate_registered(mock_distributed_accelerate):
    _ = runner.invoke(
        get_app(),
        [
            "distributed",
            "accelerate",
            "-m",
            "oumi",
            "train",
            "--config",
            "some_config",
        ],
    )
    mock_distributed_accelerate.assert_called_once()
