import sys
from collections.abc import Callable
from inspect import signature
from typing import get_type_hints
from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

from oumi.cli.distributed_run import accelerate, torchrun
from oumi.cli.env import env
from oumi.cli.evaluate import evaluate
from oumi.cli.fetch import fetch
from oumi.cli.infer import infer
from oumi.cli.judge import judge_dataset_file
from oumi.cli.launch import cancel, down, status, stop, up, which
from oumi.cli.launch import run as launcher_run
from oumi.cli.main import _get_cli_event, get_app
from oumi.cli.synth import synth
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
def mock_synth():
    with patch("oumi.cli.main.synth") as m_synth:
        _copy_command(m_synth, synth)
        yield m_synth


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
def mock_judge():
    with patch("oumi.cli.main.judge_dataset_file") as m_judge_dataset_file:
        _copy_command(m_judge_dataset_file, judge_dataset_file)
        yield m_judge_dataset_file


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


def test_main_synth_registered(mock_synth):
    _ = runner.invoke(
        get_app(), ["synth", "--config", "some/path", "--allow_extraargs"]
    )
    mock_synth.assert_called_once()


def test_main_synthesize_registered(mock_synth):
    _ = runner.invoke(
        get_app(), ["synthesize", "--config", "some/path", "--allow_extraargs"]
    )
    mock_synth.assert_called_once()


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


def test_main_env_registered(mock_env):
    _ = runner.invoke(get_app(), ["env"])
    mock_env.assert_called_once()


def test_main_judge_registered(mock_judge):
    _ = runner.invoke(
        get_app(),
        [
            "judge",
            "dataset",
            "--config",
            "./my_judge_config",
            "--input",
            "./my_input_file.jsonl",
            "--output",
            "./my_output_file.jsonl",
        ],
    )
    mock_judge.assert_called_once()


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


def test_get_cli_event_basic_command():
    """Basic command without flags."""
    with patch.object(sys, "argv", ["oumi", "train"]):
        event_name, props = _get_cli_event()
        assert event_name == "cli-train"
        assert props == {"subcommand": None, "help": False}


def test_get_cli_event_command_with_subcommand():
    """Command with subcommand (e.g., judge dataset)."""
    with patch.object(sys, "argv", ["oumi", "judge", "dataset"]):
        event_name, props = _get_cli_event()
        assert event_name == "cli-judge"
        assert props == {"subcommand": "dataset", "help": False}


def test_get_cli_event_short_help_flag_only():
    """Global help with -h flag."""
    with patch.object(sys, "argv", ["oumi", "-h"]):
        event_name, props = _get_cli_event()
        assert event_name == "cli"
        assert props == {"subcommand": None, "help": True}


def test_get_cli_event_long_help_flag_only():
    """Global help with --help flag."""
    with patch.object(sys, "argv", ["oumi", "--help"]):
        event_name, props = _get_cli_event()
        assert event_name == "cli"
        assert props == {"subcommand": None, "help": True}


def test_get_cli_event_command_with_short_help():
    """Command-specific help with -h."""
    with patch.object(sys, "argv", ["oumi", "train", "-h"]):
        event_name, props = _get_cli_event()
        assert event_name == "cli-train"
        assert props == {"subcommand": None, "help": True}


def test_get_cli_event_command_with_long_help():
    """Command-specific help with --help."""
    with patch.object(sys, "argv", ["oumi", "train", "--help"]):
        event_name, props = _get_cli_event()
        assert event_name == "cli-train"
        assert props == {"subcommand": None, "help": True}


def test_get_cli_event_subcommand_with_help():
    """Subcommand-specific help."""
    with patch.object(sys, "argv", ["oumi", "judge", "dataset", "-h"]):
        event_name, props = _get_cli_event()
        assert event_name == "cli-judge"
        assert props == {"subcommand": "dataset", "help": True}


def test_get_cli_event_no_arguments():
    """No arguments - just the binary name."""
    with patch.object(sys, "argv", ["oumi"]):
        event_name, props = _get_cli_event()
        assert event_name == "cli"
        assert props == {"subcommand": None, "help": False}


def test_get_cli_event_command_with_config_flag():
    """Command with --config flag and value."""
    with patch.object(
        sys, "argv", ["oumi", "train", "--config", "path/to/config.yaml"]
    ):
        event_name, props = _get_cli_event()
        assert event_name == "cli-train"
        assert props == {"subcommand": None, "help": False}


def test_get_cli_event_launch_subcommand():
    """Launch command with subcommand."""
    with patch.object(sys, "argv", ["oumi", "launch", "up", "--cluster", "my-cluster"]):
        event_name, props = _get_cli_event()
        assert event_name == "cli-launch"
        assert props == {"subcommand": "up", "help": False}


def test_get_cli_event_cache_subcommand():
    """Cache command with subcommand."""
    with patch.object(sys, "argv", ["oumi", "cache", "ls"]):
        event_name, props = _get_cli_event()
        assert event_name == "cli-cache"
        assert props == {"subcommand": "ls", "help": False}


def test_get_cli_event_distributed_subcommand():
    """Distributed command with subcommand and additional args."""
    with patch.object(
        sys, "argv", ["oumi", "distributed", "torchrun", "-m", "oumi", "train"]
    ):
        event_name, props = _get_cli_event()
        assert event_name == "cli-distributed"
        assert props == {"subcommand": "torchrun", "help": False}


def test_get_cli_event_flag_with_equals_syntax():
    """Flag using --flag=value syntax."""
    with patch.object(sys, "argv", ["oumi", "train", "--config=path/to/config.yaml"]):
        event_name, props = _get_cli_event()
        assert event_name == "cli-train"
        assert props == {"subcommand": None, "help": False}


def test_get_cli_event_short_flag_with_value():
    """Short flag with value (e.g., -c config.yaml)."""
    with patch.object(sys, "argv", ["oumi", "train", "-c", "llama3.1-8b"]):
        event_name, props = _get_cli_event()
        assert event_name == "cli-train"
        assert props == {"subcommand": None, "help": False}


def test_get_cli_event_global_help_before_command():
    """Global --help flag before command - captures global help intent."""
    with patch.object(sys, "argv", ["oumi", "--help", "train"]):
        event_name, props = _get_cli_event()
        # Command is None because --help comes first (global help)
        assert event_name == "cli"
        assert props == {"subcommand": None, "help": True}


def test_get_cli_event_multiple_flags_after_command():
    """Multiple flags after command."""
    with patch.object(
        sys, "argv", ["oumi", "infer", "--config", "cfg.yaml", "--interactive"]
    ):
        event_name, props = _get_cli_event()
        assert event_name == "cli-infer"
        assert props == {"subcommand": None, "help": False}


def test_get_cli_event_only_captures_first_two_positional():
    """Only the first two positional args are captured (command + subcommand)."""
    with patch.object(sys, "argv", ["oumi", "cache", "rm", "model-name"]):
        event_name, props = _get_cli_event()
        assert event_name == "cli-cache"
        # Only captures "rm" as subcommand, "model-name" is ignored
        assert props == {"subcommand": "rm", "help": False}


def test_get_cli_event_help_flag_anywhere():
    """Help flag detected regardless of position in args."""
    with patch.object(sys, "argv", ["oumi", "train", "--config", "cfg.yaml", "--help"]):
        event_name, props = _get_cli_event()
        assert event_name == "cli-train"
        assert props["help"] is True


def test_get_cli_event_double_dash_separator():
    """Double dash (--) is treated as a flag and stops positional parsing."""
    with patch.object(sys, "argv", ["oumi", "--", "train"]):
        event_name, props = _get_cli_event()
        # -- stops parsing, so no command is captured
        assert event_name == "cli"
        assert props == {"subcommand": None, "help": False}


def test_get_cli_event_command_after_double_dash():
    """Positional args after -- in normal usage."""
    with patch.object(sys, "argv", ["oumi", "train", "--", "-weird-filename"]):
        event_name, props = _get_cli_event()
        assert event_name == "cli-train"
        assert props == {"subcommand": None, "help": False}


def test_get_cli_event_both_help_flags():
    """Both -h and --help flags present."""
    with patch.object(sys, "argv", ["oumi", "train", "-h", "--help"]):
        event_name, props = _get_cli_event()
        assert event_name == "cli-train"
        assert props["help"] is True
