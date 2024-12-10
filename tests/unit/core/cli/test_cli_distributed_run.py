import copy
import logging
import sys
from unittest.mock import Mock, patch

import pytest
import typer
from typer.testing import CliRunner

from oumi.core.cli.cli_utils import CONTEXT_ALLOW_EXTRA_ARGS
from oumi.core.cli.distributed_run import accelerate, torchrun
from oumi.utils.logging import logger

runner = CliRunner()


#
# Fixtures
#
@pytest.fixture
def app():
    fake_app = typer.Typer()
    fake_app.command(context_settings=CONTEXT_ALLOW_EXTRA_ARGS)(accelerate)
    fake_app.command(context_settings=CONTEXT_ALLOW_EXTRA_ARGS)(torchrun)
    yield fake_app


@pytest.fixture
def mock_os():
    with patch("oumi.core.cli.distributed_run.os") as os_mock:
        yield os_mock


@pytest.fixture
def mock_popen():
    with patch("oumi.core.cli.distributed_run.Popen") as popen_mock:
        yield popen_mock


def test_torchrun_skypilot_single_gpu(
    app,
    mock_os,
    mock_popen,
    monkeypatch,
):
    test_env_vars = {
        "SKYPILOT_NODE_IPS": "mymachine",
        "SKYPILOT_NODE_RANK": 0,
        "SKYPILOT_NUM_GPUS_PER_NODE": 1,
    }
    mock_os.environ.copy.return_value = copy.deepcopy(test_env_vars)

    mock_process = Mock()
    mock_popen.return_value = mock_process
    mock_process.wait.return_value = 0

    monkeypatch.setattr("oumi.core.cli.distributed_run.sys.stdout", sys.stdout)
    monkeypatch.setattr("oumi.core.cli.distributed_run.sys.stderr", sys.stderr)

    _ = runner.invoke(
        app,
        [
            "torchrun",
            "-m",
            "oumi.train",
            "training.max_steps=20",
            "--log-level",
            "ERROR",
        ],
    )

    mock_popen.assert_called_once_with(
        [
            "torchrun",
            "--nnodes=1",
            "--node-rank=0",
            "--nproc-per-node=1",
            "--master-addr=mymachine",
            "--master-port=8007",
            "-m",
            "oumi.train",
            "training.max_steps=20",
        ],
        env=copy.deepcopy(test_env_vars),
        stdout=sys.stdout,
        stderr=sys.stderr,
        bufsize=1,
        universal_newlines=True,
    )
    assert logger.level == logging.ERROR


def test_torchrun_skypilot_multi_gpu(
    app,
    mock_os,
    mock_popen,
    monkeypatch,
):
    test_env_vars = {
        "SKYPILOT_NODE_IPS": "x111\nx222\nx333\n",
        "SKYPILOT_NODE_RANK": 2,
        "SKYPILOT_NUM_GPUS_PER_NODE": 4,
        # Define the redundant OUMI_ variables to activate consistency checks.
        "OUMI_TOTAL_NUM_GPUS": 12,
        "OUMI_NUM_NODES": 3,
        "OUMI_MASTER_ADDR": "x111",
    }
    mock_os.environ.copy.return_value = copy.deepcopy(test_env_vars)

    mock_process = Mock()
    mock_popen.return_value = mock_process
    mock_process.wait.return_value = 0

    monkeypatch.setattr("oumi.core.cli.distributed_run.sys.stdout", sys.stdout)
    monkeypatch.setattr("oumi.core.cli.distributed_run.sys.stderr", sys.stderr)

    _ = runner.invoke(
        app,
        [
            "torchrun",
            "-m",
            "oumi.train",
            "training.max_steps=20",
            "--log-level",
            "ERROR",
        ],
    )

    mock_popen.assert_called_once_with(
        [
            "torchrun",
            "--nnodes=3",
            "--node-rank=2",
            "--nproc-per-node=4",
            "--master-addr=x111",
            "--master-port=8007",
            "-m",
            "oumi.train",
            "training.max_steps=20",
        ],
        env=copy.deepcopy(test_env_vars),
        stdout=sys.stdout,
        stderr=sys.stderr,
        bufsize=1,
        universal_newlines=True,
    )
    assert logger.level == logging.ERROR
