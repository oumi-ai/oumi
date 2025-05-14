import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, NamedTuple, Optional

import pytest
import yaml

from oumi.core.configs import TrainingConfig
from oumi.core.configs.params.training_params import TrainerType
from oumi.utils.io_utils import load_json
from oumi.utils.torch_utils import device_cleanup
from tests import get_configs_dir
from tests.e2e import get_e2e_test_output_dir, is_file_not_empty
from tests.markers import requires_gpus


class TrainTestConfig(NamedTuple):
    test_name: str
    config_path: Path
    max_steps: int
    is_lora: bool = False
    skip: bool = False
    interactive_logs: bool = True

    trainer_type: Optional[TrainerType] = None
    model_max_length: Optional[int] = None
    batch_size: Optional[int] = None
    gradient_accumulation_steps: Optional[int] = None
    dataloader_num_workers: Optional[int] = None
    dataloader_prefetch_factor: Optional[int] = None
    save_steps: Optional[int] = None
    save_final_model: Optional[bool] = None
    enable_wandb: Optional[bool] = False  # Disable `wandb`` by default


def _check_checkpoint_dir(
    dir_path: Path, *, is_lora: bool, validate_extra_files: bool = False
):
    """Helper to verify model directory structure."""
    # Check essential model files
    essential_files = [
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "trainer_state.json",
        "training_args.bin",
    ]
    if is_lora:
        essential_files = ["adapter_config.json"] + essential_files  # OPE-938
    else:
        essential_files = ["config.json", "generation_config.json"] + essential_files

    for file in essential_files:
        assert (dir_path / file).is_file(), f"Missing {file} in {dir_path}"
        assert is_file_not_empty(dir_path / file), f"Empty {file} in {dir_path}"

    model_basename = "adapter_model" if is_lora else "model"
    model_safetensors = dir_path / f"{model_basename}.safetensors"

    if model_safetensors.exists():
        assert model_safetensors.is_file(), (
            f"Exists but not a file: {model_safetensors}"
        )
        assert is_file_not_empty(model_safetensors), f"Empty {model_safetensors}"
    else:
        # The model is sharded. Let's validate model shards.
        model_index_json = dir_path / f"{model_basename}.safetensors.index.json"
        assert model_index_json.is_file(), (
            f"{model_basename} safetensors missing: "
            f"None of {model_index_json} and {model_safetensors} exists"
        )
        model_shards = list(
            sorted(dir_path.glob(f"{model_basename}-*-of-*.safetensors"))
        )
        assert len(model_shards) > 0, (
            f"No '{model_basename}-*-of-*.safetensors' files found under {dir_path}"
        )
        for model_shard in model_shards:
            assert (model_shard).is_file(), f"Missing {model_shard}"
            assert is_file_not_empty(model_shard), f"Empty {model_shard}"
        index_dict: dict[str, Any] = load_json(model_index_json)
        assert "weight_map" in index_dict, f"No `weights_map` in {model_index_json}"
        assert isinstance(index_dict["weight_map"], dict)
        index_shards = {
            (dir_path / shard) for shard in set(index_dict["weight_map"].values())
        }
        assert index_shards == set(model_shards), (
            "Shards defined in model index are inconsistent with shards on file system"
        )

    if is_lora:
        config = load_json(dir_path / "adapter_config.json")
        for key in ("peft_type", "r", "target_modules", "lora_alpha"):
            assert key in config, f"Invalid model config: Missing '{key}'\n{config}"
    else:  # OPE-938
        # Verify config.json is valid JSON
        config = load_json(dir_path / "config.json")
        assert "model_type" in config, f"Invalid model config:\n{config}"

        # Verify generation config
        gen_config = load_json(dir_path / "generation_config.json")
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

    if validate_extra_files:
        # Additional checkpoint-specific files
        checkpoint_files = ["scheduler.pt"]
        for file in checkpoint_files:
            assert (dir_path / file).exists(), f"Missing {file} in checkpoint"
            assert is_file_not_empty(dir_path / file), f"Empty {file} in checkpoint"

        optimizer_files = ["optimizer.pt", "optimizer.bin"]
        num_valid_optimizer_files = 0
        for file in optimizer_files:
            optimizer_file = dir_path / file
            if optimizer_file.exists():
                assert is_file_not_empty(optimizer_file), (
                    f"Empty {optimizer_file} in checkpoint"
                )
                num_valid_optimizer_files += 1
        assert num_valid_optimizer_files == 1, (
            f"Exactly one of {optimizer_files} must exist. "
            f"Got: {num_valid_optimizer_files}"
        )

        if (dir_path / "rng_state.pth").exists():
            assert is_file_not_empty(dir_path / "rng_state.pth")
        else:
            rng_state_shards = list(sorted(dir_path.glob("rng_state_*.pth")))
            assert len(rng_state_shards) > 1
            for file in rng_state_shards:
                assert is_file_not_empty(dir_path / file), f"Empty {file} in checkpoint"


def get_train_test_id_fn(val):
    assert isinstance(val, TrainTestConfig), f"{type(val)}: {val}"
    return val.test_name


def _test_train_impl(
    test_config: TrainTestConfig,
    tmp_path: Path,
    *,
    use_distributed: bool,
    cleanup_output_dir_on_success: bool = True,
    telemetry_callback_enabled: bool = True,
):
    device_cleanup()
    if test_config.skip:
        pytest.skip(f"Skipped the test '{test_config.test_name}'!")
        return

    interactive_logs = test_config.interactive_logs

    test_tag = f"[{test_config.test_name}]"

    _START_TIME = time.perf_counter()
    output_dir = get_e2e_test_output_dir(test_config.test_name, tmp_path=tmp_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Copy config file to output directory
        assert test_config.config_path.exists(), (
            f"{test_tag} Path doesn't exist: {test_config.config_path}"
        )
        assert test_config.config_path.is_file(), (
            f"{test_tag} Path is not a file: {test_config.config_path}"
        )

        # Verify the config is loadable
        try:
            TrainingConfig.from_yaml(test_config.config_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load training config from: {test_config.config_path}"
            ) from e

        assert test_config.max_steps > 0, f"max_steps: {test_config.max_steps}"

        cmd: list[str] = []
        if use_distributed:
            cmd.append("oumi distributed torchrun -m oumi train")
        else:
            cmd.append("oumi train")

        # Execute training command
        cmd.extend(
            [
                "-c",
                str(test_config.config_path),
                "--training.max_steps",
                str(test_config.max_steps),
                "--training.output_dir",
                str(output_dir / "train"),
                "--training.run_name",
                test_config.test_name,
            ]
        )

        for param_name, param_value in [
            ("model_max_length", test_config.model_max_length),
        ]:
            if param_value is not None:
                cmd.append(f"--model.{param_name}={str(param_value)}")

        for param_name, param_value in [
            ("trainer_type", test_config.trainer_type),
            ("per_device_train_batch_size", test_config.batch_size),
            ("gradient_accumulation_steps", test_config.gradient_accumulation_steps),
            ("dataloader_num_workers", test_config.dataloader_num_workers),
            ("dataloader_prefetch_factor", test_config.dataloader_prefetch_factor),
            ("save_steps", test_config.save_steps),
            ("save_final_model", test_config.save_final_model),
            ("enable_wandb", test_config.enable_wandb),
        ]:
            if param_value is not None:
                cmd.append(f"--training.{param_name}={str(param_value)}")

        env_vars = dict(os.environ)
        if "TOKENIZERS_PARALLELISM" not in env_vars:
            # Resolves the warning: "Avoid using `tokenizers` before the fork ..."
            env_vars["TOKENIZERS_PARALLELISM"] = "false"

        shell_command = " ".join(cmd)
        print(f"{test_tag} Running the command:\n{shell_command}\n")
        device_cleanup()
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
            print(
                f"{test_tag} Training job successfully finished in {duration_sec:.2f}s!"
            )
        else:
            print(
                f"{test_tag} Training job failed with error code: {result.returncode} "
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
        assert train_output_dir.is_dir(), (
            f"{test_tag} Output directory is not a directory"
        )

        # If saving is disabled, then return early.
        if (test_config.save_steps is not None and test_config.save_steps <= 0) and (
            test_config.save_final_model is not None
            and not test_config.save_final_model
        ):
            return

        # Check main output directory structure
        _check_checkpoint_dir(train_output_dir, is_lora=test_config.is_lora)

        # Verify checkpoint directory
        checkpoints = list(train_output_dir.glob("checkpoint-*"))
        assert len(checkpoints) > 0, f"{test_tag} No checkpoints found"

        for checkpoint in checkpoints:
            _check_checkpoint_dir(
                checkpoint, is_lora=test_config.is_lora, validate_extra_files=True
            )

        # Check logs directory
        logs_dir = train_output_dir / "logs"
        assert logs_dir.exists(), f"{test_tag} Logs directory not found"
        rank_logs = list(logs_dir.glob("rank_*.log"))
        num_ranks = len(rank_logs)
        assert num_ranks > 0, f"{test_tag} No rank logs found"
        for idx in range(num_ranks):
            assert is_file_not_empty(rank_logs[idx]), (
                f"{test_tag} Empty rank log file: {rank_logs[idx]}"
            )

        # Check telemetry directory
        telemetry_dir = train_output_dir / "telemetry"
        assert telemetry_dir.exists(), f"{test_tag} Telemetry directory not found"
        assert telemetry_dir.is_dir(), (
            f"{test_tag} Telemetry directory  is not a directory"
        )

        telemetry_files = [
            "devices_info.txt",
            "training_config.yaml",
            "world_size.json",
        ] + (
            [
                "telemetry_callback_metrics_rank0000.json",
                "telemetry_callback_rank0000.json",
            ]
            if telemetry_callback_enabled
            else []
        )

        for file in telemetry_files:
            file_path = telemetry_dir / file
            assert file_path.exists(), f"Missing telemetry file: {file}"
            assert is_file_not_empty(file_path), f"Empty telemetry file: {file}"

        # Verify telemetry content
        with open(telemetry_dir / "training_config.yaml") as f:
            training_config = yaml.safe_load(f)
            assert "model" in training_config, (
                f"{test_tag} Invalid training config: {training_config}"
            )
            assert "training" in training_config, (
                f"{test_tag} Invalid training config: {training_config}"
            )

        with open(telemetry_dir / "world_size.json") as f:
            world_size = json.load(f)
            assert "WORLD_SIZE" in world_size
            assert world_size.get("WORLD_SIZE", None) == num_ranks, (
                f"{test_tag} World size is inconsistent with: {num_ranks}"
            )

    except Exception as e:
        duration_sec = time.perf_counter() - _START_TIME
        print(f"{test_tag} Test failed: {str(e)}")
        print(f"{test_tag} Test duration: {duration_sec:.2f}s")
        print(f"{test_tag} Test artifacts can be found in: {output_dir}")
        raise

    if cleanup_output_dir_on_success:
        # Clean-up temp data to stay under disk quota.
        print(f"{test_tag} Cleaning up output dir on success: '{output_dir}'...")
        shutil.rmtree(output_dir)


@requires_gpus(count=1, min_gb=24.0)
@pytest.mark.parametrize(
    "test_config",
    [
        TrainTestConfig(
            test_name="train_text_llama_1b",
            config_path=(
                get_configs_dir()
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
            test_name="pretrain_text_fineweb",
            config_path=(
                get_configs_dir()
                / "examples"
                / "fineweb_ablation_pretraining"
                / "ddp"
                / "train.yaml"
            ),
            batch_size=2,
            max_steps=5,
            model_max_length=512,
        ),
        TrainTestConfig(
            test_name="pretrain_text_gpt2",
            config_path=(
                get_configs_dir() / "recipes" / "gpt2" / "pretraining" / "train.yaml"
            ),
            batch_size=16,
            dataloader_num_workers=2,
            dataloader_prefetch_factor=4,
            max_steps=20,
        ),
        TrainTestConfig(
            test_name="train_text_smollm_135m_sft",
            config_path=(
                get_configs_dir() / "recipes" / "smollm" / "sft" / "135m" / "train.yaml"
            ),
            max_steps=10,
        ),
    ],
    ids=get_train_test_id_fn,
)
@pytest.mark.e2e
@pytest.mark.single_gpu
def test_train_text_1gpu_24gb(
    test_config: TrainTestConfig,
    tmp_path: Path,
):
    _test_train_impl(test_config=test_config, tmp_path=tmp_path, use_distributed=False)


@requires_gpus(count=4, min_gb=39.0)
@pytest.mark.parametrize(
    "test_config",
    [
        TrainTestConfig(
            test_name="train_text_qwen3_30b_a3b_trl_sft_lora",
            config_path=(
                get_configs_dir()
                / "recipes"
                / "qwen3"
                / "sft"
                / "30b_a3b_lora"
                / "train.yaml"
            ),
            trainer_type=TrainerType.TRL_SFT,
            max_steps=5,
            save_steps=5,
            is_lora=True,
        ),
    ],
    ids=get_train_test_id_fn,
)
@pytest.mark.e2e
@pytest.mark.multi_gpu
def test_train_text_4gpu_40gb(test_config: TrainTestConfig, tmp_path: Path):
    _test_train_impl(
        test_config=test_config,
        tmp_path=tmp_path,
        use_distributed=True,
    )


@requires_gpus(count=4, min_gb=39.0)
@pytest.mark.parametrize(
    "test_config",
    [
        TrainTestConfig(
            test_name="train_mm_qwen2_vl_2b_trl_sft_fft",
            config_path=(
                get_configs_dir()
                / "recipes"
                / "vision"
                / "qwen2_vl_2b"
                / "sft"
                / "full"
                / "train.yaml"
            ),
            trainer_type=TrainerType.TRL_SFT,
            max_steps=5,
            save_steps=5,
        ),
        TrainTestConfig(
            test_name="train_mm_qwen2_vl_2b_oumi_fft",
            config_path=(
                get_configs_dir()
                / "recipes"
                / "vision"
                / "qwen2_vl_2b"
                / "sft"
                / "full"
                / "train.yaml"
            ),
            trainer_type=TrainerType.OUMI,
            max_steps=5,
            save_steps=0,
            save_final_model=False,
        ),
    ],
    ids=get_train_test_id_fn,
)
@pytest.mark.e2e
@pytest.mark.multi_gpu
def test_train_multimodal_4gpu_40gb(test_config: TrainTestConfig, tmp_path: Path):
    _test_train_impl(
        test_config=test_config,
        tmp_path=tmp_path,
        use_distributed=True,
    )


@requires_gpus(count=1, min_gb=39.0)
@pytest.mark.parametrize(
    "test_config",
    [
        TrainTestConfig(
            test_name="train_mm_qwen2_vl_2b_trl_sft_lora",
            config_path=(
                get_configs_dir()
                / "recipes"
                / "vision"
                / "qwen2_vl_2b"
                / "sft"
                / "lora"
                / "train.yaml"
            ),
            trainer_type=TrainerType.TRL_SFT,
            max_steps=5,
            save_steps=5,
            is_lora=True,
        ),
    ],
    ids=get_train_test_id_fn,
)
@pytest.mark.e2e
@pytest.mark.single_gpu
def test_train_multimodal_lora_1gpu_40gb(test_config: TrainTestConfig, tmp_path: Path):
    _test_train_impl(
        test_config=test_config,
        tmp_path=tmp_path,
        use_distributed=False,
    )


@requires_gpus(count=4, min_gb=79.0)
@pytest.mark.parametrize(
    "test_config",
    [
        TrainTestConfig(
            test_name="train_mm_llama3_2_vision_11b_full",
            config_path=(
                get_configs_dir()
                / "recipes"
                / "vision"
                / "llama3_2_vision"
                / "sft"
                / "11b_full"
                / "train.yaml"
            ),
            max_steps=5,
            save_steps=5,
        ),
        TrainTestConfig(
            test_name="train_mm_llama3_2_vision_11b_lora",
            config_path=(
                get_configs_dir()
                / "recipes"
                / "vision"
                / "llama3_2_vision"
                / "sft"
                / "11b_lora"
                / "train.yaml"
            ),
            is_lora=True,
            max_steps=5,
            save_steps=5,
        ),
        TrainTestConfig(
            test_name="train_mm_llava_7b_sft_full",
            config_path=(
                get_configs_dir()
                / "recipes"
                / "vision"
                / "llava_7b"
                / "sft"
                / "train.yaml"
            ),
            max_steps=5,
            save_steps=5,
        ),
    ],
    ids=get_train_test_id_fn,
)
@pytest.mark.e2e
@pytest.mark.multi_gpu
def test_train_multimodal_fsdp_4gpu_80gb(test_config: TrainTestConfig, tmp_path: Path):
    _test_train_impl(
        test_config=test_config,
        tmp_path=tmp_path,
        use_distributed=True,
    )


@requires_gpus(count=4, min_gb=39.0)
@pytest.mark.parametrize(
    "test_config",
    [
        TrainTestConfig(
            test_name="train_grpo_tldr_qwen2_500m",
            config_path=(get_configs_dir() / "examples" / "grpo_tldr" / "train.yaml"),
            max_steps=3,
            save_steps=3,
        ),
        TrainTestConfig(
            test_name="train_grpo_letter_counting",
            config_path=(
                get_configs_dir()
                / "examples"
                / "letter_counting"
                / "grpo"
                / "train.yaml"
            ),
            max_steps=3,
            save_steps=3,
        ),
    ],
    ids=get_train_test_id_fn,
)
@pytest.mark.e2e
@pytest.mark.multi_gpu
def test_train_grpo_4gpu_40gb(test_config: TrainTestConfig, tmp_path: Path):
    _test_train_impl(
        test_config=test_config,
        tmp_path=tmp_path,
        use_distributed=True,
        telemetry_callback_enabled=False,
    )
