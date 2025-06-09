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
from typing import Any, Callable, Final, Optional, Union

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
    trainer_type: TrainerType,
    metrics_function: Optional[Callable],
    reward_functions: list[Callable],
    collator: Optional[Callable],
    additional_trainer_kwargs: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if trainer_type == TrainerType.OUMI:
        kwargs["config"] = config

    if trainer_type in (TrainerType.TRL_GRPO, TrainerType.VERL_GRPO):
        if metrics_function:
            raise ValueError(f"metrics_function isn't supported for {trainer_type}")
        if collator:
            raise ValueError(f"collator isn't supported for {trainer_type}")
        kwargs["reward_funcs"] = reward_functions
    else:
        kwargs["compute_metrics"] = metrics_function
        kwargs["data_collator"] = collator
    kwargs.update(additional_trainer_kwargs or {})
    return kwargs


def _log_feedback_request():
    """Logs a feedback request for the platform."""
    logger.info(
        "\n\n» We're always looking for feedback. "
        "What's one thing we can improve? https://oumi.ai/feedback"
    )


def _verl_train(
    partial_trainer: Callable[[], BaseTrainer], checkpoint_location: Optional[str]
):
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
    def _run_verl_train(
        partial_trainer: Callable[[], BaseTrainer], checkpoint_location: Optional[str]
    ):
        trainer = partial_trainer()
        trainer.train(resume_from_checkpoint=checkpoint_location)

        logger.info("Training is Complete.")

    ray.get(_run_verl_train.remote(partial_trainer, checkpoint_location))
    _log_feedback_request()


def train(
    config: TrainingConfig,
    additional_model_kwargs: Optional[dict[str, Any]] = None,
    additional_trainer_kwargs: Optional[dict[str, Any]] = None,
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

    config = _finalize_training_config(config)

    if is_local_process_zero():
        logger.info(f"TrainingConfig:\n{pformat(config)}")
        if telemetry_dir and is_world_process_zero():
            config.to_yaml(str(telemetry_dir / "training_config.yaml"))

    # Initialize tokenizer and processor.
    tokenizer: Optional[BaseTokenizer] = None
    if is_custom_model(config.model.model_name) and not config.model.tokenizer_name:
        # Keep tokenizer as None for custom models unless `tokenizer_name` is specified.
        tokenizer = None
    else:
        tokenizer = build_tokenizer(config.model)

    processor: Optional[BaseProcessor] = None
    if is_image_text_llm(config.model):
        assert tokenizer is not None, (
            "Tokenizer can't be None because all VLM-s are non-custom currently"
        )
        # Only create `processor` for VLM-s for now.
        processor = build_processor(
            config.model.model_name,
            tokenizer,
            trust_remote_code=config.model.trust_remote_code,
            processor_kwargs=config.model.processor_kwargs,
        )
        # Setting remove_unused_columns to False is needed for VLM training with the
        # TRL_SFT trainer.
        # See: https://huggingface.co/docs/trl/en/sft_trainer#training-the-vision-language-model
        # Otherwise, SFTTrainer's overridden `_set_signature_columns_if_needed()`
        # function will result in columns needed for VLM training (e.g. `pixel_values`)
        # to be dropped from the dataset.
        if config.training.trainer_type == TrainerType.TRL_SFT:
            config.training.trainer_kwargs["remove_unused_columns"] = False

    # Load datasets.
    train_dataset = build_dataset_mixture(
        config.data,
        tokenizer,
        DatasetSplit.TRAIN,
        seq_length=config.model.model_max_length,
    )

    eval_dataset = None
    if len(config.data.get_split(DatasetSplit.VALIDATION).datasets) != 0:
        eval_dataset = build_dataset_mixture(
            config.data,
            tokenizer,
            DatasetSplit.VALIDATION,
            seq_length=config.model.model_max_length,
        )

    trainer_type: Final[TrainerType] = config.training.trainer_type
    metrics_function: Optional[Callable] = build_metrics_function(config.training)
    reward_functions: list[Callable] = build_reward_functions(config.training)
    if trainer_type == TrainerType.TRL_GRPO:
        if len(reward_functions) == 0:
            logger.warning(f"No reward_function specified for {trainer_type}!")
        if not isinstance(train_dataset, BaseExperimentalGrpoDataset) and isinstance(
            train_dataset, (hf_datasets.Dataset, hf_datasets.IterableDataset)
        ):
            train_dataset = try_prepare_trl_grpo_dataset(train_dataset)
        if (
            eval_dataset is not None
            and not isinstance(eval_dataset, BaseExperimentalGrpoDataset)
            and isinstance(
                eval_dataset, (hf_datasets.Dataset, hf_datasets.IterableDataset)
            )
        ):
            eval_dataset = try_prepare_trl_grpo_dataset(eval_dataset)

    collator: Optional[Callable] = build_collator_from_config(config, tokenizer)

    training_kwargs = _create_optional_training_kwargs(
        config,
        trainer_type,
        metrics_function,
        reward_functions,
        collator,
        additional_trainer_kwargs=additional_trainer_kwargs,
    )

    checkpoint_location = _find_checkpoint_to_resume_from(
        config.training.resume_from_checkpoint,
        config.training.try_resume_from_last_checkpoint,
        config.training.output_dir,
    )

    # verl training is handled separately because:
    # 1. It uses Ray
    # 2. Some of the setup below is not applicable.
    if config.training.trainer_type == TrainerType.VERL_GRPO:
        create_trainer_fn = build_trainer(trainer_type, processor=processor)

        # We don't initialize the trainer here because it needs to run in a remote Ray
        # function.
        partial_trainer = functools.partial(
            create_trainer_fn,
            processing_class=tokenizer,
            config=config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processor=processor,
            **training_kwargs,
        )
        _verl_train(partial_trainer, checkpoint_location)
        return

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

    use_peft = config.training.use_peft and config.peft

    # Build model.
    model = build_model(
        model_params=config.model,
        peft_params=config.peft if use_peft else None,
        **(additional_model_kwargs or {}),
    )

    if use_peft:
        logger.info("Building PEFT model...")
        model = build_peft_model(
            model, config.training.enable_gradient_checkpointing, config.peft
        )

    if is_local_process_zero():
        log_number_of_model_parameters(model)
        if config.training.log_model_summary:
            log_model_summary(
                model, telemetry_dir / "model_summary.txt" if telemetry_dir else None
            )

    # trl's SFTTrainer has its own dataset processing code. We should skip it if
    # the dataset is already processed, i.e. it's tokenized and has an `input_ids`
    # field. This generally occurs if the dataset is:
    # 1. In the Oumi registry and thus is processed by the `BasePretrainingDataset` or
    # `BaseSftDataset` classes
    # 2. Packing is requested, and thus is processed by the
    # `PretrainingAsyncTextDataset` class
    # See OPE-1108 for more details.
    if config.training.trainer_type == TrainerType.TRL_SFT:
        example = next(iter(train_dataset))
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
    create_trainer_fn: Callable[..., BaseTrainer] = build_trainer(
        trainer_type, processor=processor
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
                processing_class=tokenizer,
                args=config.training,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                callbacks=callbacks,
                **training_kwargs,
            )

        with torch.profiler.record_function("log_and_verify"):
            log_nvidia_gpu_runtime_info(log_prefix="GPU Metrics Before Training:")
            verify_torch_distributed_initialized_if_needed()

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
    _log_feedback_request()
