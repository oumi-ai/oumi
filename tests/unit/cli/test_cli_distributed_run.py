import copy
import logging
import sys
import tempfile
from unittest.mock import Mock, patch

import pytest
import typer
from typer.testing import CliRunner

from oumi.cli.cli_utils import CONTEXT_ALLOW_EXTRA_ARGS
from oumi.cli.distributed_run import accelerate, torchrun
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
    with patch("oumi.cli.distributed_run.os") as os_mock:
        yield os_mock


@pytest.fixture
def mock_popen():
    with patch("oumi.cli.distributed_run.Popen") as popen_mock:
        yield popen_mock


@pytest.fixture
def mock_torch():
    torch_mock = Mock()
    with patch.dict("sys.modules", {"torch": torch_mock}):
        yield torch_mock


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

    monkeypatch.setattr("oumi.cli.distributed_run.sys.stdout", sys.stdout)
    monkeypatch.setattr("oumi.cli.distributed_run.sys.stderr", sys.stderr)

    _ = runner.invoke(
        app,
        [
            "torchrun",
            "-m",
            "oumi",
            "train",
            "--training.max_steps",
            "20",
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
            "oumi",
            "train",
            "--training.max_steps",
            "20",
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

    monkeypatch.setattr("oumi.cli.distributed_run.sys.stdout", sys.stdout)
    monkeypatch.setattr("oumi.cli.distributed_run.sys.stderr", sys.stderr)

    _ = runner.invoke(
        app,
        [
            "torchrun",
            "-m",
            "oumi",
            "train",
            "--training.max_steps",
            "20",
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
            "oumi",
            "train",
            "--training.max_steps",
            "20",
        ],
        env=copy.deepcopy(test_env_vars),
        stdout=sys.stdout,
        stderr=sys.stderr,
        bufsize=1,
        universal_newlines=True,
    )
    assert logger.level == logging.ERROR


def test_torchrun_polaris_multi_gpu(
    app,
    mock_os,
    mock_popen,
    monkeypatch,
):
    with tempfile.NamedTemporaryFile("w+t") as file_nodelist:
        # file_nodelist.name
        file_nodelist.writelines(["z111\n", "x222\n", "x333\n"])
        file_nodelist.flush()

        test_env_vars = {
            "PBS_NODEFILE": file_nodelist.name,
            "PMI_RANK": 1,
            "PBS_JOBID": "123456.polaris",
            # Define the redundant OUMI_ variables to activate consistency checks.
            "OUMI_TOTAL_NUM_GPUS": 12,
            "OUMI_NUM_NODES": 3,
            "OUMI_MASTER_ADDR": "z111",
        }
        mock_os.environ.copy.return_value = copy.deepcopy(test_env_vars)

        mock_process = Mock()
        mock_popen.return_value = mock_process
        mock_process.wait.return_value = 0

        monkeypatch.setattr("oumi.cli.distributed_run.sys.stdout", sys.stdout)
        monkeypatch.setattr("oumi.cli.distributed_run.sys.stderr", sys.stderr)

        _ = runner.invoke(
            app,
            [
                "torchrun",
                "-m",
                "oumi",
                "train",
                "--training.max_steps",
                "21",
                "--log-level",
                "DEBUG",
            ],
        )

        mock_popen.assert_called_once()
        mock_popen.assert_called_once_with(
            [
                "torchrun",
                "--nnodes=3",
                "--node-rank=1",
                "--nproc-per-node=4",
                "--master-addr=z111",
                "--master-port=8007",
                "-m",
                "oumi",
                "train",
                "--training.max_steps",
                "21",
            ],
            env=copy.deepcopy(test_env_vars),
            stdout=sys.stdout,
            stderr=sys.stderr,
            bufsize=1,
            universal_newlines=True,
        )
        assert logger.level == logging.DEBUG


def test_torchrun_frontier_multi_gpu(
    app,
    mock_os,
    mock_popen,
    monkeypatch,
):
    test_env_vars = {
        "SLURM_NODELIST": "z111,x222,x333",
        "PMI_RANK": 1,
        "SLURM_JOBID": "123456.frontier",
        # Define the redundant OUMI_ variables to activate consistency checks.
        "OUMI_TOTAL_NUM_GPUS": 24,
        "OUMI_NUM_NODES": 3,
        "OUMI_MASTER_ADDR": "z111",
    }
    mock_os.environ.copy.return_value = copy.deepcopy(test_env_vars)

    mock_process = Mock()
    mock_popen.return_value = mock_process
    mock_process.wait.return_value = 0

    monkeypatch.setattr("oumi.cli.distributed_run.sys.stdout", sys.stdout)
    monkeypatch.setattr("oumi.cli.distributed_run.sys.stderr", sys.stderr)

    _ = runner.invoke(
        app,
        [
            "torchrun",
            "-m",
            "oumi",
            "train",
            "--training.max_steps",
            "21",
            "--log-level",
            "DEBUG",
        ],
    )

    mock_popen.assert_called_once()
    mock_popen.assert_called_once_with(
        [
            "torchrun",
            "--nnodes=3",
            "--node-rank=1",
            "--nproc-per-node=8",
            "--master-addr=z111",
            "--master-port=8007",
            "-m",
            "oumi",
            "train",
            "--training.max_steps",
            "21",
        ],
        env=copy.deepcopy(test_env_vars),
        stdout=sys.stdout,
        stderr=sys.stderr,
        bufsize=1,
        universal_newlines=True,
    )
    assert logger.level == logging.DEBUG


def test_accelerate_skypilot_multi_gpu(
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

    monkeypatch.setattr("oumi.cli.distributed_run.sys.stdout", sys.stdout)
    monkeypatch.setattr("oumi.cli.distributed_run.sys.stderr", sys.stderr)

    _ = runner.invoke(
        app,
        [
            "accelerate",
            "launch",
            "-m",
            "oumi",
            "evaluate",
            "--log-level",
            "DEBUG",
        ],
    )

    mock_popen.assert_called_once_with(
        [
            "accelerate",
            "launch",
            "--num_machines=3",
            "--machine_rank=2",
            "--num_processes=12",
            "--main_process_ip=x111",
            "--main_process_port=8007",
            "-m",
            "oumi",
            "evaluate",
        ],
        env=copy.deepcopy(test_env_vars),
        stdout=sys.stdout,
        stderr=sys.stderr,
        bufsize=1,
        universal_newlines=True,
    )
    assert logger.level == logging.DEBUG


def test_torchrun_localmachine_multi_gpu(
    app,
    mock_os,
    mock_popen,
    mock_torch,
    monkeypatch,
):
    test_env_vars = {
        # No environment vars set
    }
    mock_os.environ.copy.return_value = copy.deepcopy(test_env_vars)
    mock_torch.cuda.device_count.return_value = 8

    mock_process = Mock()
    mock_popen.return_value = mock_process
    mock_process.wait.return_value = 0

    monkeypatch.setattr("oumi.cli.distributed_run.sys.stdout", sys.stdout)
    monkeypatch.setattr("oumi.cli.distributed_run.sys.stderr", sys.stderr)

    _ = runner.invoke(
        app,
        [
            "torchrun",
            "-m",
            "oumi",
            "train",
            "--training.max_steps",
            "20",
            "--log-level",
            "ERROR",
        ],
    )

    mock_popen.assert_called_once_with(
        [
            "torchrun",
            "--nnodes=1",
            "--node-rank=0",
            "--nproc-per-node=8",
            "--master-addr=127.0.0.1",
            "--master-port=8007",
            "-m",
            "oumi",
            "train",
            "--training.max_steps",
            "20",
        ],
        env=copy.deepcopy(test_env_vars),
        stdout=sys.stdout,
        stderr=sys.stderr,
        bufsize=1,
        universal_newlines=True,
    )
    assert logger.level == logging.ERROR


def test_torchrun_localmachine_multi_gpu_masteraddress(
    app,
    mock_os,
    mock_popen,
    mock_torch,
    monkeypatch,
):
    test_env_vars = {"MASTER_ADDRESS": "111.0.0.0", "MASTER_PORT": 1337}
    mock_os.environ.copy.return_value = copy.deepcopy(test_env_vars)
    mock_torch.cuda.device_count.return_value = 8

    mock_process = Mock()
    mock_popen.return_value = mock_process
    mock_process.wait.return_value = 0

    monkeypatch.setattr("oumi.cli.distributed_run.sys.stdout", sys.stdout)
    monkeypatch.setattr("oumi.cli.distributed_run.sys.stderr", sys.stderr)

    _ = runner.invoke(
        app,
        [
            "torchrun",
            "-m",
            "oumi",
            "train",
            "--training.max_steps",
            "20",
            "--log-level",
            "ERROR",
        ],
    )

    mock_popen.assert_called_once_with(
        [
            "torchrun",
            "--nnodes=1",
            "--node-rank=0",
            "--nproc-per-node=8",
            "--master-addr=111.0.0.0",
            "--master-port=1337",
            "-m",
            "oumi",
            "train",
            "--training.max_steps",
            "20",
        ],
        env=copy.deepcopy(test_env_vars),
        stdout=sys.stdout,
        stderr=sys.stderr,
        bufsize=1,
        universal_newlines=True,
    )
    assert logger.level == logging.ERROR
