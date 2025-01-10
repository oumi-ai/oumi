import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import NamedTuple, Optional

import pytest
import yaml

from oumi.core.configs import TrainingConfig
from oumi.utils.io_utils import get_oumi_root_directory
from tests.markers import requires_gpus

CONFIG_FOLDER_ROOT = get_oumi_root_directory().parent.parent.resolve() / "configs"


def _get_output_dir(test_name: str, tmp_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if os.environ.get("OUMI_E2E_TESTS_OUTPUT_DIR"):
        output_base = Path(os.environ["OUMI_E2E_TESTS_OUTPUT_DIR"])
    else:
        output_base = tmp_path / "e2e_tests"

    return output_base / f"{timestamp}_{test_name}"


def _is_file_not_empty(file_path: Path) -> bool:
    """Check if a file is not empty."""
    return file_path.stat().st_size > 0


def _check_checkpoint_dir(dir_path: Path):
    """Helper to verify model directory structure."""
    # Check essential model files
    essential_files = [
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "trainer_state.json",
        "training_args.bin",
    ]
    for file in essential_files:
        assert (dir_path / file).is_file(), f"Missing {file} in {dir_path}"
        assert _is_file_not_empty(dir_path / file), f"Empty {file} in {dir_path}"

    # Verify config.json is valid JSON
    with open(dir_path / "config.json") as f:
        config = json.load(f)
        assert "model_type" in config, "Invalid model config"

    # Verify generation config
    with open(dir_path / "generation_config.json") as f:
        gen_config = json.load(f)
        assert isinstance(gen_config, dict), "Invalid generation config"

    # Verify special tokens map
    with open(dir_path / "special_tokens_map.json") as f:
        tokens_map = json.load(f)
        assert isinstance(tokens_map, dict), "Invalid special tokens map"

    # Verify tokenizer config
    with open(dir_path / "tokenizer_config.json") as f:
        tok_config = json.load(f)
        assert isinstance(tok_config, dict), "Invalid tokenizer config"

    # Verify tokenizer
    with open(dir_path / "tokenizer.json") as f:
        tokenizer = json.load(f)
        assert isinstance(tokenizer, dict), "Invalid tokenizer file"

    # Verify trainer state
    with open(dir_path / "trainer_state.json") as f:
        trainer_state = json.load(f)
        assert "best_model_checkpoint" in trainer_state, "Invalid trainer state"
        assert "log_history" in trainer_state, "Missing training logs in trainer state"


class TrainTestConfig(NamedTuple):
    test_name: str
    config_path: Path
    max_steps: int
    model_max_length: Optional[int] = None
    save_steps: Optional[int] = None


def get_train_test_id_fn(val):
    assert isinstance(val, TrainTestConfig), f"{type(val)}: {val}"
    return val.test_name


@requires_gpus(count=1, min_gb=24.0)
@pytest.mark.parametrize(
    "test_config",
    [
        TrainTestConfig(
            test_name="train_llama_1b",
            config_path=(
                CONFIG_FOLDER_ROOT
                / "recipes"
                / "llama3_2"
                / "sft"
                / "1b_full"
                / "train.yaml"
            ),
            max_steps=10,
            model_max_length=128,
        ),
        TrainTestConfig(
            test_name="train_qwen2_vl_2b",
            config_path=(
                CONFIG_FOLDER_ROOT
                / "recipes"
                / "vision"
                / "qwen2_vl_2b"
                / "sft"
                / "train.yaml"
            ),
            max_steps=5,
            save_steps=5,
        ),
    ],
    ids=get_train_test_id_fn,
)
@pytest.mark.skip(reason="Skipping until the markers are configured")
def test_train(
    test_config: TrainTestConfig, tmp_path: Path, interactive_logs: bool = True
):
    _START_TIME = time.perf_counter()
    test_tag = f"[{test_config.test_name}]"
    output_dir = _get_output_dir(test_config.test_name, tmp_path=tmp_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Copy config file to output directory
        assert (
            test_config.config_path.exists()
        ), f"{test_tag} Path doesn't exist: {test_config.config_path}"
        assert (
            test_config.config_path.is_file()
        ), f"{test_tag} Path is not a file: {test_config.config_path}"

        # Verify the config is loadable
        try:
            TrainingConfig.from_yaml(test_config.config_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load training config from: {test_config.config_path}"
            ) from e

        assert test_config.max_steps > 0, f"max_steps: {test_config.max_steps}"

        # Execute training command
        cmd = [
            "oumi train",
            "-c",
            str(test_config.config_path),
            "--training.max_steps",
            str(test_config.max_steps),
            "--training.output_dir",
            str(output_dir / "train"),
        ]
        if (
            test_config.model_max_length is not None
            and test_config.model_max_length > 0
        ):
            cmd.extend(
                [
                    "--model.model_max_length",
                    str(test_config.model_max_length),
                ]
            )

        if test_config.save_steps is not None:
            cmd.extend(
                [
                    "--training.save_steps",
                    str(test_config.save_steps),
                ]
            )

        env_vars = dict(os.environ)
        if "TOKENIZERS_PARALLELISM" not in env_vars:
            # Resolves the warning: "Avoid using `tokenizers` before the fork ..."
            env_vars["TOKENIZERS_PARALLELISM"] = "false"

        shell_command = " ".join(cmd)
        print(f"{test_tag} Running the command:\n{shell_command}\n")
        result = subprocess.run(
            shell_command,
            shell=True,
            text=True,
            capture_output=(not interactive_logs),
            stdout=(sys.stdout if interactive_logs else None),
            stderr=(sys.stderr if interactive_logs else None),
            env=env_vars,
        )
        duration_sec = time.perf_counter() - _START_TIME
        if result.returncode == 0:
            print(f"{test_tag} Successfully finished in {duration_sec:.2f}s!")
        else:
            print(
                f"{test_tag} Training failed with error code: {result.returncode} "
                f"in {duration_sec:.2f}s!"
            )
            if not interactive_logs:
                print(f"{test_tag} STDOUT:\n\n{result.stdout}\n\n")
                print(f"{test_tag} STDERR:\n\n{result.stderr}\n\n")
            assert result.returncode == 0, (
                f"{test_tag} Training failed with error code: {result.returncode}"
                + ("" if interactive_logs else f"\nSTDERR:\n\n{result.stderr}\n")
            )

        # Check output directory exists
        train_output_dir = output_dir / "train"
        assert train_output_dir.exists(), f"{test_tag} Output directory was not created"
        assert (
            train_output_dir.is_dir()
        ), f"{test_tag} Output directory is not a directory"

        # Check main output directory structure
        _check_checkpoint_dir(train_output_dir)

        # Verify checkpoint directory
        checkpoints = list(train_output_dir.glob("checkpoint-*"))
        assert len(checkpoints) > 0, f"{test_tag} No checkpoints found"

        for checkpoint in checkpoints:
            _check_checkpoint_dir(checkpoint)

            # Additional checkpoint-specific files
            checkpoint_files = ["optimizer.pt", "rng_state.pth", "scheduler.pt"]
            for file in checkpoint_files:
                assert (checkpoint / file).exists(), f"Missing {file} in checkpoint"
                assert _is_file_not_empty(
                    checkpoint / file
                ), f"{test_tag} Empty {file} in checkpoint"

        # Check logs directory
        logs_dir = train_output_dir / "logs"
        assert logs_dir.exists(), f"{test_tag} Logs directory not found"
        rank_logs = list(logs_dir.glob("rank_*.log"))
        assert len(rank_logs) > 0, f"{test_tag} No rank logs found"
        assert _is_file_not_empty(rank_logs[0]), f"{test_tag} Empty rank log file"

        # Check telemetry directory
        telemetry_dir = train_output_dir / "telemetry"
        assert telemetry_dir.exists(), f"{test_tag} Telemetry directory not found"
        assert (
            telemetry_dir.is_dir()
        ), f"{test_tag} Telemetry directory  is not a directory"

        telemetry_files = [
            "devices_info.txt",
            "telemetry_callback_metrics_rank0000.json",
            "telemetry_callback_rank0000.json",
            "training_config.yaml",
            "world_size.json",
        ]

        for file in telemetry_files:
            file_path = telemetry_dir / file
            assert file_path.exists(), f"Missing telemetry file: {file}"
            assert _is_file_not_empty(file_path), f"Empty telemetry file: {file}"

        # Verify telemetry content
        with open(telemetry_dir / "training_config.yaml") as f:
            training_config = yaml.safe_load(f)
            assert (
                "model" in training_config
            ), f"{test_tag} Invalid training config: {training_config}"
            assert (
                "training" in training_config
            ), f"{test_tag} Invalid training config: {training_config}"

        with open(telemetry_dir / "world_size.json") as f:
            world_size = json.load(f)
            assert (
                world_size.get("WORLD_SIZE", None) == 1
            ), f"{test_tag} Invalid world size format"

    except Exception as e:
        print(f"{test_tag} Test failed: {str(e)}")
        print(f"{test_tag} Test artifacts can be found in: {output_dir}")
        raise
