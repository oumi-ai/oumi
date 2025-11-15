# Copyright 2025 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import time
from importlib.metadata import version
from pathlib import Path
from pprint import pformat
from typing import Any, Callable, Final, Optional, Union, cast

import datasets as hf_datasets
import torch
import transformers
from transformers.trainer_utils import get_last_checkpoint

from oumi.builders import (
    build_collator_from_config,
    build_dataset_mixture,
    build_metrics_function,
    build_model,
    build_peft_model,
    build_processor,
    build_reward_functions,
    build_rollout_function,
    build_tokenizer,
    build_trainer,
    build_training_callbacks,
    is_image_text_llm,
)
from oumi.core.configs import (
    DatasetSplit,
    TrainerType,
    TrainingConfig,
)
from oumi.core.configs.internal.supported_models import (
    is_custom_model,
)
from oumi.core.datasets import BaseExperimentalGrpoDataset
from oumi.core.distributed import (
    barrier,
    cleanup_distributed,
    get_device_rank_info,
    init_distributed,
    is_distributed,
    is_local_process_zero,
    is_world_process_zero,
    prepare_accelerate_fsdp_run,
    verify_torch_distributed_initialized_if_needed,
)
from oumi.core.processors.base_processor import BaseProcessor
from oumi.core.tokenizers import BaseTokenizer
from oumi.core.trainers import BaseTrainer, LocalTrainer, VerlGrpoTrainer
from oumi.performance.torch_profiler_utils import torch_profile
from oumi.utils.device_utils import (
    log_nvidia_gpu_runtime_info,
)
from oumi.utils.distributed_utils import is_using_accelerate, is_using_accelerate_fsdp
from oumi.utils.git_utils import get_git_revision_hash, get_git_tag
from oumi.utils.grpo_utils import try_prepare_trl_grpo_dataset
from oumi.utils.io_utils import save_json
from oumi.utils.logging import configure_logger, logger
from oumi.utils.torch_utils import (
    coerce_model_to_dtype,
    device_cleanup,
    get_torch_dtype,
    log_devices_info,
    log_model_summary,
    log_number_of_model_parameters,
    log_peak_gpu_memory,
    log_versioning_info,
)
from oumi.utils.version_utils import is_dev_build


def _find_checkpoint_to_resume_from(
    resume_from_checkpoint: Optional[str],
    try_resume_from_last_checkpoint: bool,
    output_dir: str,
) -> Optional[str]:
    """Finds and returns the last checkpoint path to be passed to Trainer."""
    checkpoint_path = None
    if resume_from_checkpoint:
        checkpoint_path = resume_from_checkpoint
    elif try_resume_from_last_checkpoint:
        checkpoint_path = get_last_checkpoint(output_dir)
        if not checkpoint_path:
            logger.warning(f"No checkpoints found under {output_dir}")

    if checkpoint_path:
        logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
        return checkpoint_path
    return None


def _ensure_dir_exists(output_dir: Union[str, Path], human_readable_name: str) -> None:
    if not output_dir:
        raise ValueError(f"{human_readable_name} is not specified!")
    output_dir_path: Path = Path(output_dir)
    if output_dir_path.exists():
        if not output_dir_path.is_dir():
            raise ValueError(
                f"{human_readable_name}='{output_dir}' is not a directory!"
            )
    elif is_local_process_zero():
        logger.info(f"Creating {human_readable_name}: {output_dir}...")
        output_dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Created {human_readable_name} "
            f"absolute path: {str(output_dir_path.absolute())}"
        )


def _create_training_dirs(config: TrainingConfig) -> None:
    """Creates misc directories referenced in config."""
    _ensure_dir_exists(config.training.output_dir, "training.output_dir")
    telemetry_dir = config.training.telemetry_dir
    if telemetry_dir:
        _ensure_dir_exists(telemetry_dir, "training.telemetry_dir")


def _log_training_info(config: TrainingConfig) -> None:
    """Logs misc infos about training config/devices/etc. Writes to files."""
    telemetry_dir = config.training.telemetry_dir
    if telemetry_dir and is_world_process_zero():
        device_rank_info = get_device_rank_info()
        save_json(
            {
                "LOCAL_WORLD_SIZE": device_rank_info.local_world_size,
                "WORLD_SIZE": device_rank_info.world_size,
            },
            telemetry_dir / "world_size.json",
        )

    if is_local_process_zero():
        log_versioning_info()
        log_devices_info(
            (telemetry_dir / "devices_info.txt")
            if telemetry_dir and is_world_process_zero()
            else None
        )
        oumi_version = version("oumi")
        logger.info(f"Oumi version: {oumi_version}")
        if is_dev_build():
            logger.info(f"Git revision hash: {get_git_revision_hash()}")
            logger.info(f"Git tag: {get_git_tag()}")


def _log_feedback_request():
    """Logs a feedback request for the platform."""
    logger.info(
        "\n\n» We're always looking for feedback. "
        "What's one thing we can improve? https://oumi.ai/feedback"
    )


def _verl_train(partial_trainer: Callable[[], BaseTrainer]):
    """Runs verl training.

    This function initializes Ray, and then initializes and kicks off the trainer in a
    remote Ray function.
    """
    try:
        import ray  # pyright: ignore[reportMissingImports]
    except ModuleNotFoundError:
        raise RuntimeError(
            "ray is not installed. Please install it with `pip install 'oumi[gpu]'`."
        )
    if not ray.is_initialized():
        logger.info("Initializing Ray cluster...")
        ray.init(
            runtime_env={
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": "true",
                    "NCCL_DEBUG": "WARN",
                    "VLLM_LOGGING_LEVEL": "WARN",
                }
            }
        )

    # We define the remote function as a sub function so that the `@ray.remote`
    # decorator is only run if this function is run. This function should only be run
    # if ray is installed, preventing errors when it isn't.
    @ray.remote
    def _run_verl_train(partial_trainer: Callable[[], BaseTrainer]):
        trainer = partial_trainer()
        trainer.train()

        logger.info("Training is Complete.")

    ray.get(_run_verl_train.remote(partial_trainer))
    _log_feedback_request()


def train(
    config: TrainingConfig,
    additional_model_kwargs: Optional[dict[str, Any]] = None,
    additional_trainer_kwargs: Optional[dict[str, Any]] = None,
    verbose: bool = False,
) -> None:
    """Trains a model using the provided configuration."""
    _START_TIME = time.time()

    _create_training_dirs(config)
    _log_training_info(config)

    # Configure logging to file
    log_dir = Path(config.training.output_dir) / "logs"
    for logger_name in ("oumi", "oumi.telemetry"):
        configure_logger(logger_name, level=config.training.log_level, log_dir=log_dir)

    telemetry_dir = config.training.telemetry_dir

    if is_local_process_zero():
        if verbose:
            logger.info(f"TrainingConfig:\n{pformat(config)}")
        if telemetry_dir and is_world_process_zero():
            config_path = telemetry_dir / "training_config.yaml"
            config.to_yaml(str(config_path))
            logger.info(f"Training config saved to {config_path}")

    # TODO: Move to config post_init
    if is_image_text_llm(config.model):
        # Setting remove_unused_columns to False is needed for VLM training with the
        # TRL_SFT trainer.
        # See: https://huggingface.co/docs/trl/en/sft_trainer#training-the-vision-language-model
        # Otherwise, SFTTrainer's overridden `_set_signature_columns_if_needed()`
        # function will result in columns needed for VLM training (e.g. `pixel_values`)
        # to be dropped from the dataset.
        if config.training.trainer_type == TrainerType.TRL_SFT:
            config.training.trainer_kwargs["remove_unused_columns"] = False
            logger.info(
                "Set `training.trainer_kwargs.remove_unused_columns=False` for VLM "
                "training with TRL_SFT trainer."
            )

    # TODO: Split this off
    # verl training is handled separately because:
    # 1. It uses Ray
    # 2. Some of the setup below is not applicable.
    # if config.training.trainer_type == TrainerType.VERL_GRPO:
    #     # We don't initialize the trainer here because it needs to run in a remote Ray
    #     # function.
    #     partial_trainer = functools.partial(
    #         VerlGrpoTrainer,
    #         processing_class=tokenizer,
    #         config=config,
    #         train_dataset=train_dataset,
    #         eval_dataset=eval_dataset,
    #         processor=processor,
    #         **training_kwargs,
    #     )
    #     _verl_train(partial_trainer)
    #     return

    if is_distributed():
        init_distributed(timeout_minutes=config.training.nccl_default_timeout_minutes)

    # We support running FSDP Oumi training without being invoked from the Accelerate
    # launcher. We detect this with the following:
    # 1. Accelerate's environment variables aren't set
    # 2. We are running with a HF-family trainer (HF, TRL_SFT, TRL_DPO, TRL_GRPO)
    # 3. FSDP is enabled in the Oumi config
    # In this case, we mimic an Accelerate launcher run by setting the necessary
    # environment variables.
    # Note that normal Accelerate launcher runs won't be affected.
    if (
        not is_using_accelerate()
        and config.training.trainer_type != TrainerType.OUMI
        and config.fsdp.enable_fsdp
    ):
        accelerate_env_vars = prepare_accelerate_fsdp_run(config)
        logger.info(
            f"Set Accelerate environment variables for FSDP: {accelerate_env_vars}"
        )

    # Reclaim memory before training starts.
    device_cleanup()

    with torch_profile(
        config.training.profiler,
        training_output_dir=config.training.output_dir,
        record_function_name="oumi.train",
    ) as profiler:
        with torch.profiler.record_function("create_trainer"):
            trainer = LocalTrainer.from_config(
                config, additional_model_kwargs, additional_trainer_kwargs, profiler
            )

        with torch.profiler.record_function("log_and_verify"):
            log_nvidia_gpu_runtime_info(log_prefix="GPU Metrics Before Training:")
            verify_torch_distributed_initialized_if_needed()

        with torch.profiler.record_function("wait_for_all_ranks"):
            # Make sure all workers start training at the same time.
            barrier()

        # Should we move this inside the train() function?
        checkpoint_location = _find_checkpoint_to_resume_from(
            config.training.resume_from_checkpoint,
            config.training.try_resume_from_last_checkpoint,
            config.training.output_dir,
        )
        with torch.profiler.record_function("train"):
            logger.info(f"Training init time: {time.time() - _START_TIME:.3f}s")
            logger.info(
                f"Starting training... "
                f"({config.training.trainer_type}, "
                f"transformers: {transformers.__version__})"
            )
            trainer.train(resume_from_checkpoint=checkpoint_location)

    logger.info("Training is Complete.")

    log_nvidia_gpu_runtime_info(log_prefix="GPU Metrics After Training:")
    log_peak_gpu_memory()

    # Save final checkpoint & training state.
    if config.training.save_final_model:
        logger.info("Saving final state...")
        trainer.save_state()

        barrier()

        logger.info("Saving final model...")

        trainer.save_model(config=config)

    barrier()

    if is_distributed():
        cleanup_distributed()
    _log_feedback_request()
