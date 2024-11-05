import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

import pytest
import yaml

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


@requires_gpus(count=1)
@pytest.mark.skip(reason="Skipping until the markers are configured")
def test_train_llama_1b(tmp_path: Path):
    test_name = "train_llama_1b"

    output_dir = _get_output_dir(test_name, tmp_path=tmp_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Copy config file to output directory
        config_path = (
            CONFIG_FOLDER_ROOT
            / "recipes"
            / "llama3_2"
            / "sft"
            / "1b_full"
            / "train.yaml"
        )

        # Execute training command
        cmd = [
            "oumi train",
            "-c",
            str(config_path),
            "--training.max_steps",
            "10",
            "--training.output_dir",
            str(output_dir / "train"),
            "--model.model_max_length",
            "128",
        ]

        result = subprocess.run(
            " ".join(cmd), shell=True, capture_output=True, text=True
        )

        # Check command executed successfully
        assert result.returncode == 0, f"Training failed with error: {result.stderr}"

        # Check output directory exists
        train_output_dir = output_dir / "train"
        assert train_output_dir.exists(), "Output directory was not created"

        # Check main output directory structure
        _check_checkpoint_dir(train_output_dir)

        # Verify checkpoint directory
        checkpoints = list(train_output_dir.glob("checkpoint-*"))
        assert len(checkpoints) > 0, "No checkpoints found"

        for checkpoint in checkpoints:
            _check_checkpoint_dir(checkpoint)

            # Additional checkpoint-specific files
            checkpoint_files = ["optimizer.pt", "rng_state.pth", "scheduler.pt"]
            for file in checkpoint_files:
                assert (checkpoint / file).exists(), f"Missing {file} in checkpoint"
                assert _is_file_not_empty(
                    checkpoint / file
                ), f"Empty {file} in checkpoint"

        # Check logs directory
        logs_dir = train_output_dir / "logs"
        assert logs_dir.exists(), "Logs directory not found"
        rank_logs = list(logs_dir.glob("rank_*.log"))
        assert len(rank_logs) > 0, "No rank logs found"
        assert _is_file_not_empty(rank_logs[0]), "Empty rank log file"

        # Check telemetry directory
        telemetry_dir = train_output_dir / "telemetry"
        assert telemetry_dir.exists(), "Telemetry directory not found"

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
            assert "model" in training_config, "Invalid training config"
            assert "training" in training_config, "Invalid training config"

        with open(telemetry_dir / "world_size.json") as f:
            world_size = json.load(f)
            assert world_size.get("WORLD_SIZE", None) == 1, "Invalid world size format"

    except Exception as e:
        print(f"Test failed: {str(e)}")
        print(f"Test artifacts can be found in: {output_dir}")
        raise
