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

import time
from importlib.metadata import version
from pathlib import Path
from pprint import pformat
from typing import Any, Callable, Final, Optional, Union

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
from oumi.core.distributed import (
    barrier,
    cleanup_distributed,
    estimate_dataloader_num_workers,
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
from oumi.core.trainers import BaseTrainer
from oumi.performance.torch_profiler_utils import torch_profile
from oumi.utils.device_utils import (
    log_nvidia_gpu_runtime_info,
)
from oumi.utils.distributed_utils import is_using_accelerate, is_using_accelerate_fsdp
from oumi.utils.git_utils import get_git_revision_hash, get_git_tag
from oumi.utils.io_utils import save_json
from oumi.utils.logging import configure_logger, logger
from oumi.utils.torch_utils import (
    coerce_model_to_dtype,
    device_cleanup,
    get_torch_dtype,
    log_devices_info,
    log_model_summary,
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
    """Creates misc directoris referenced in config."""
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


def _finalize_training_config(config: TrainingConfig) -> TrainingConfig:
    """Updates TrainingConfig using dynamic/runtime info."""
    if config.training.dataloader_num_workers == "auto":
        # Resolve "auto" to an actual number.
        num_workers = estimate_dataloader_num_workers()
        logger.info(
            "Resolved 'training.dataloader_num_workers=auto' to "
            f"'training.dataloader_num_workers={num_workers}'"
        )
        config.training.dataloader_num_workers = num_workers

    assert isinstance(config.training.dataloader_num_workers, int)

    if config.training.trainer_type == TrainerType.TRL_GRPO:
        world_size = get_device_rank_info().world_size
        batch_size = config.training.per_device_train_batch_size
        global_batch_size = world_size * batch_size
        num_generations = config.training.grpo.num_generations
        if num_generations is not None and global_batch_size % num_generations != 0:
            logger.warning(
                f"For {config.training.trainer_type}, "
                f"global batch size ({global_batch_size}) should be evenly divisible "
                f"by `grpo.num_generations` ({num_generations}). It's not! "
                f"World size: {world_size}. "
                f"Per-device batch size: {batch_size}."
            )

    return config


def _create_optional_training_kwargs(
    config: TrainingConfig,
    tokenizer: Optional[BaseTokenizer],
    trainer_type: TrainerType,
    metrics_function: Optional[Callable],
    reward_functions: list[Callable],
    collator: Optional[Callable],
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"processing_class": tokenizer}
    if trainer_type == TrainerType.OUMI:
        kwargs["config"] = config

    if trainer_type != TrainerType.TRL_GRPO:
        kwargs["compute_metrics"] = metrics_function
        kwargs["data_collator"] = collator
    else:
        assert trainer_type == TrainerType.TRL_GRPO
        if metrics_function:
            raise ValueError(f"metrics_function isn't supported for {trainer_type}")
        if collator:
            raise ValueError(f"collator isn't supported for {trainer_type}")
        kwargs["reward_funcs"] = reward_functions
    return kwargs


def train(config: TrainingConfig, **kwargs) -> None:
    """Trains a model using the provided configuration."""
    _START_TIME = time.time()

    if is_distributed():
        init_distributed(timeout_minutes=config.training.nccl_default_timeout_minutes)

    _create_training_dirs(config)
    _log_training_info(config)

    # Configure logging to file
    log_dir = Path(config.training.output_dir) / "logs"
    for logger_name in ("oumi", "oumi.telemetry"):
        configure_logger(logger_name, level=config.training.log_level, log_dir=log_dir)

    telemetry_dir = config.training.telemetry_dir

    config = _finalize_training_config(config)

    if is_local_process_zero():
        logger.info(f"TrainingConfig:\n{pformat(config)}")
        if telemetry_dir and is_world_process_zero():
            config.to_yaml(str(telemetry_dir / "training_config.yaml"))

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

    # Initialize model and tokenizer.
    tokenizer: Optional[BaseTokenizer] = None
    if is_custom_model(config.model.model_name) and not config.model.tokenizer_name:
        # Keep tokenizer as None for custom models unless `tokenizer_name` is specified.
        tokenizer = None
    else:
        tokenizer = build_tokenizer(config.model)

    processor: Optional[BaseProcessor] = None
    if is_image_text_llm(config.model):
        assert (
            tokenizer is not None
        ), "Tokenizer can't be None because all VLM-s are non-custom currently"
        # Only create `processor` for VLM-s for now.
        processor = build_processor(
            config.model.model_name,
            tokenizer,
            trust_remote_code=config.model.trust_remote_code,
        )

    use_peft = config.training.use_peft and config.peft

    # Build model.
    model = build_model(
        model_params=config.model,
        peft_params=config.peft if use_peft else None,
        *kwargs,
    )

    if use_peft:
        logger.info("Building PEFT model...")
        model = build_peft_model(
            model, config.training.enable_gradient_checkpointing, config.peft
        )

    if config.training.log_model_summary and is_local_process_zero():
        log_model_summary(
            model, telemetry_dir / "model_summary.txt" if telemetry_dir else None
        )

    # Load data & preprocessing
    dataset = build_dataset_mixture(config, tokenizer, DatasetSplit.TRAIN)
    # return dataset
    eval_dataset = None
    if len(config.data.get_split(DatasetSplit.VALIDATION).datasets) != 0:
        eval_dataset = build_dataset_mixture(config, tokenizer, DatasetSplit.VALIDATION)

    # trl's SFTTrainer has its own dataset processing code. We should skip it if
    # the dataset is already processed, i.e. it's tokenized and has an `input_ids`
    # field. This generally occurs if the dataset is:
    # 1. In the Oumi registry and thus is processed by the `BasePretrainingDataset` or
    # `BaseSftDataset` classes
    # 2. Packing is requested, and thus is processed by the
    # `PretrainingAsyncTextDataset` class
    # See OPE-1108 for more details.
    if config.training.trainer_type == TrainerType.TRL_SFT:
        example = next(iter(dataset))
        if "input_ids" in example:
            logger.info(
                "Skipping dataset preparation for TRL_SFT trainer since the dataset is "
                "already processed."
            )
            if "dataset_kwargs" not in config.training.trainer_kwargs:
                config.training.trainer_kwargs["dataset_kwargs"] = {}
            # Skip preparing dataset if `skip_prepare_dataset` isn't already set.
            if (
                "skip_prepare_dataset"
                not in config.training.trainer_kwargs["dataset_kwargs"]
            ):
                config.training.trainer_kwargs["dataset_kwargs"][
                    "skip_prepare_dataset"
                ] = True

    # Train model
    trainer_type: Final[TrainerType] = config.training.trainer_type
    create_trainer_fn: Callable[..., BaseTrainer] = build_trainer(
        trainer_type, processor=processor
    )

    metrics_function: Optional[Callable] = build_metrics_function(config.training)
    reward_functions: list[Callable] = build_reward_functions(config.training)
    if trainer_type == TrainerType.TRL_GRPO and len(reward_functions) == 0:
        logger.warning(f"No reward_function specified for {trainer_type}!")

    collator: Optional[Callable] = build_collator_from_config(config, tokenizer)

    training_kwargs = _create_optional_training_kwargs(
        config,
        tokenizer,
        trainer_type,
        metrics_function,
        reward_functions,
        collator,
    )

    # Reclaim memory before training starts.
    device_cleanup()

    with torch_profile(
        config.training.profiler,
        training_output_dir=config.training.output_dir,
        record_function_name="oumi.train",
    ) as profiler:
        with torch.profiler.record_function("create_trainer"):
            callbacks = build_training_callbacks(config, model, profiler)

            trainer = create_trainer_fn(
                model=model,
                args=config.training,
                train_dataset=dataset,
                eval_dataset=eval_dataset,
                callbacks=callbacks,
                **training_kwargs,
            )

        with torch.profiler.record_function("log_and_verify"):
            log_nvidia_gpu_runtime_info(log_prefix="GPU Metrics Before Training:")
            verify_torch_distributed_initialized_if_needed()

        with torch.profiler.record_function("find_checkpoint_to_resume_from"):
            checkpoint_location = _find_checkpoint_to_resume_from(
                config.training.resume_from_checkpoint,
                config.training.try_resume_from_last_checkpoint,
                config.training.output_dir,
            )

        # TODO: OPE-577 - Remove when the issue is resolved.
        # QLoRA FSDP training currently has an issue where some submodules of the model
        # are float32 instead of the requested dtype. As a workaround, we coerce all
        # modules to the desired dtype. See:
        # https://github.com/huggingface/accelerate/issues/1620#issuecomment-2407102051
        if is_using_accelerate_fsdp() and config.peft.q_lora:
            # https://huggingface.co/docs/bitsandbytes/main/en/fsdp_qlora#quantized-data-storage
            quant_storage_dtype = get_torch_dtype(config.peft.bnb_4bit_quant_storage)
            if quant_storage_dtype != config.model.torch_dtype:
                raise ValueError(
                    f"BnB 4-bit quantization storage dtype must match model dtype. "
                    f"Instead got {config.peft.bnb_4bit_quant_storage} and "
                    f"{config.model.torch_dtype}."
                )
            coerce_model_to_dtype(model, config.model.torch_dtype)
            logger.info(f"Coerced model to dtype {config.model.torch_dtype}!")

        with torch.profiler.record_function("wait_for_all_ranks"):
            # Make sure all workers start training at the same time.
            barrier()

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
    logger.info(
        "\n\n» We're always looking for feedback. "
        "What's one thing we can improve? https://oumi.ai/feedback"
    )
