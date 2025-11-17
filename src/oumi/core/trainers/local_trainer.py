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
from oumi.core.trainers import BaseTrainer, VerlGrpoTrainer
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
import pathlib
from typing import Optional, cast

import peft
import transformers

from oumi.core.configs import TrainingConfig
from oumi.core.configs.params.peft_params import PeftSaveMode
from oumi.core.distributed import is_world_process_zero
from oumi.core.trainers.base_trainer import BaseTrainer
from oumi.utils.logging import logger


def _create_optional_training_kwargs(
    config: TrainingConfig,
    trainer_type: TrainerType,
    metrics_function: Optional[Callable],
    reward_functions: list[Callable],
    rollout_function: Optional[Callable],
    collator: Optional[Callable],
    processor: Optional[BaseProcessor],
    additional_trainer_kwargs: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    # Pass config to all trainer types so DeepSpeed can be configured in HF trainers
    kwargs["training_config"] = config

    if trainer_type in (TrainerType.TRL_GRPO, TrainerType.VERL_GRPO):
        if metrics_function:
            raise ValueError(f"metrics_function isn't supported for {trainer_type}")
        if collator:
            raise ValueError(f"collator isn't supported for {trainer_type}")
        kwargs["reward_funcs"] = reward_functions
        kwargs["rollout_func"] = rollout_function
    else:
        kwargs["compute_metrics"] = metrics_function
        kwargs["data_collator"] = collator

    # Handle GKD teacher model - pass the teacher model path to the trainer constructor
    # TRL's GKDTrainer will load the teacher model automatically using
    # teacher_model_init_kwargs from the config
    if trainer_type == TrainerType.TRL_GKD:
        if config.training.gkd.teacher_model_name_or_path:
            kwargs["teacher_model"] = config.training.gkd.teacher_model_name_or_path

    if trainer_type == TrainerType.OUMI:
        kwargs["processor"] = processor

    kwargs.update(additional_trainer_kwargs or {})
    return kwargs


class LocalTrainer(BaseTrainer):
    def __init__(
        self,
        model,
        tokenizer,
        config,
        train_dataset,
        eval_dataset,
        callbacks,
        training_kwargs,
    ):
        """Initializes a local trainer."""
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.callbacks = callbacks
        self.training_kwargs = training_kwargs

    @staticmethod
    def init_training():
        # TODO: Init distributed, etc.
        # Need to determine what needs to be run before object creation, and what needs
        # to be run before training
        pass

    @classmethod
    def from_config(
        cls,
        config: TrainingConfig,
        additional_model_kwargs,
        additional_trainer_kwargs,
        profiler,
    ) -> "LocalTrainer":
        """Creates a local trainer from a configuration."""
        # Initialize tokenizer and processor.
        tokenizer: Optional[BaseTokenizer] = None
        if is_custom_model(config.model.model_name) and not config.model.tokenizer_name:
            # Keep tokenizer as None for custom models unless `tokenizer_name` is specified.
            tokenizer = None
        else:
            tokenizer = build_tokenizer(config.model)

        processor: Optional[BaseProcessor] = None
        # Only create `processor` for VLM-s for now.
        if is_image_text_llm(config.model):
            assert tokenizer is not None, (
                "Tokenizer can't be None because all VLM-s are non-custom currently"
            )
            processor = build_processor(
                config.model.model_name,
                tokenizer,
                trust_remote_code=config.model.trust_remote_code,
                processor_kwargs=config.model.processor_kwargs,
            )

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
        rollout_function: Optional[Callable] = build_rollout_function(config.training)
        if trainer_type == TrainerType.TRL_GRPO:
            if len(reward_functions) == 0:
                logger.warning(f"No reward_function specified for {trainer_type}!")
            if not isinstance(
                train_dataset, BaseExperimentalGrpoDataset
            ) and isinstance(
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

        collator: Optional[Callable] = build_collator_from_config(
            config, tokenizer, debug=config.training.log_examples
        )

        # Build model.
        use_peft = config.training.use_peft and config.peft
        model = build_model(
            model_params=config.model,
            peft_params=config.peft if use_peft else None,
            **(additional_model_kwargs or {}),
        )
        # TODO: Move this to build_model
        if is_local_process_zero():
            log_number_of_model_parameters(model)
            if config.training.log_model_summary:
                log_model_summary(
                    model,
                    config.training.telemetry_dir / "model_summary.txt"
                    if config.training.telemetry_dir
                    else None,
                )

        if use_peft:
            logger.info("Building PEFT model...")
            model = build_peft_model(
                model, config.training.enable_gradient_checkpointing, config.peft
            )

        # TODO: Move this to build_peft_model
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
            if config.model.torch_dtype_str == "auto":
                raise ValueError(
                    "torch_dtype cannot be 'auto' for QLoRA FSDP training. "
                    "Please specify a dtype."
                )
            coerce_model_to_dtype(model, cast(torch.dtype, config.model.torch_dtype))
            logger.info(f"Coerced model to dtype {config.model.torch_dtype}!")

        training_kwargs = _create_optional_training_kwargs(
            config,
            trainer_type,
            metrics_function,
            reward_functions,
            rollout_function,
            collator,
            processor,
            additional_trainer_kwargs=additional_trainer_kwargs,
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

        callbacks = build_training_callbacks(config, model, profiler)
        return cls(
            model=model,
            tokenizer=tokenizer,
            config=config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=callbacks,
            training_kwargs=training_kwargs,
        )
