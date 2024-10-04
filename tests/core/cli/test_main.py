from inspect import signature
from typing import Callable
from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

from oumi.core.cli.evaluate import evaluate
from oumi.core.cli.infer import infer
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


def test_main_evaluate_registered(mock_eval):
    _ = runner.invoke(
        get_app(), ["evaluate", "--config", "some/path", "--allow_extra" "args"]
    )
    mock_eval.assert_called_once()
