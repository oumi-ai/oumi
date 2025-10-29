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
from typing import Any, Callable, Optional

import torch

from oumi.builders import (
    build_collator_from_config,
    build_dataset_mixture,
    # build_metrics_functions,
    build_model,
    build_processor,
    build_tokenizer,
    build_trainer,
    build_tuner,
    is_image_text_llm,
)
from oumi.builders.callbacks import build_training_callbacks
from oumi.builders.models import build_peft_model
from oumi.core.configs import (
    DatasetSplit,
    TuningConfig,
)
from oumi.core.configs.internal.supported_models import (
    is_custom_model,
)
from oumi.core.configs.params.peft_params import PeftParams
from oumi.core.configs.params.training_params import TrainerType, TrainingParams
from oumi.core.configs.training_config import TrainingConfig
from oumi.core.distributed import (
    barrier,
    cleanup_distributed,
    get_device_rank_info,
    is_distributed,
    is_local_process_zero,
    is_world_process_zero,
    verify_torch_distributed_initialized_if_needed,
)
from oumi.core.processors.base_processor import BaseProcessor
from oumi.core.tokenizers import BaseTokenizer
from oumi.core.trainers.hf_trainer import HuggingFaceTrainer
from oumi.performance.torch_profiler_utils import torch_profile
from oumi.train import _ensure_dir_exists
from oumi.utils.device_utils import (
    log_nvidia_gpu_runtime_info,
)
from oumi.utils.git_utils import get_git_revision_hash, get_git_tag
from oumi.utils.io_utils import save_json
from oumi.utils.logging import configure_logger, logger
from oumi.utils.torch_utils import (
    device_cleanup,
    log_devices_info,
    log_model_summary,
    log_number_of_model_parameters,
    log_peak_gpu_memory,
    log_versioning_info,
)
from oumi.utils.version_utils import is_dev_build


def _create_tuning_dirs(config: TuningConfig) -> None:
    """Creates misc directories referenced in config."""
    _ensure_dir_exists(config.tuning.output_dir, "tuning.output_dir")
    telemetry_dir = config.tuning.telemetry_dir
    if telemetry_dir:
        _ensure_dir_exists(telemetry_dir, "training.telemetry_dir")


def _log_tuning_info(config: TuningConfig) -> None:
    """Logs misc infos about training config/devices/etc. Writes to files."""
    telemetry_dir = config.tuning.telemetry_dir
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


def tune(
    config: TuningConfig,
    additional_model_kwargs: Optional[dict[str, Any]] = None,
    additional_tuner_kwargs: Optional[dict[str, Any]] = None,
    verbose: bool = False,
) -> None:
    """Tunes a model using the provided configuration."""
    _START_TIME = time.time()

    _create_tuning_dirs(config)
    _log_tuning_info(config)

    # Configure logging to file
    log_dir = Path(config.tuning.output_dir) / "logs"
    for logger_name in ("oumi", "oumi.telemetry"):
        configure_logger(
            logger_name,
            level=config.tuning.log_level,
            log_dir=log_dir,
        )

    telemetry_dir = config.tuning.telemetry_dir
    if is_local_process_zero():
        if verbose:
            logger.info(f"TuningConfig:\n{pformat(config)}")
        if telemetry_dir and is_world_process_zero():
            config_path = telemetry_dir / "tuning_config.yaml"
            config.to_yaml(str(config_path))
            logger.info(f"Training config saved to {config_path}")

    # Initialize tokenizer
    tokenizer: Optional[BaseTokenizer] = None
    if is_custom_model(config.model.model_name) and not config.model.tokenizer_name:
        # Keep tokenizer as None for custom models unless `tokenizer_name` is specified.
        tokenizer = None
    else:
        tokenizer = build_tokenizer(config.model)

    # Initialize processor if needed
    processor: Optional[BaseProcessor] = None
    if is_image_text_llm(config.model):
        assert tokenizer is not None
        processor = build_processor(
            config.model.model_name,
            tokenizer,
            trust_remote_code=config.model.trust_remote_code,
            processor_kwargs=config.model.processor_kwargs,
        )

    # Load datasets
    train_dataset = build_dataset_mixture(
        config.data,
        tokenizer,
        DatasetSplit.TRAIN,
        seq_length=config.model.model_max_length,
    )

    # Eval dataset is required for hyperparameter tuning
    eval_dataset = build_dataset_mixture(
        config.data,
        tokenizer,
        DatasetSplit.VALIDATION,
        seq_length=config.model.model_max_length,
    )

    # metrics_functions = build_metrics_functions(config.tuning)
    tuner = build_tuner(config.tuning)

    def objective(
        suggested_training_params: dict[str, Any],
        suggested_peft_params: dict[str, Any],
        trial_number: int,
    ) -> dict[str, float]:
        """Objective function for hyperparameter tuning."""
        # Merge suggested training params with fixed params
        training_params = {
            **config.tuning.fixed_training_params,
            **suggested_training_params,
        }
        training_params["output_dir"] = str(
            Path(
                config.tuning.output_dir,
                f"trial_{trial_number}",
            )
        )
        trial_training_params = TrainingParams(**training_params)

        # Merged suggested and fixed PEFT params
        peft_params = {
            **config.tuning.fixed_peft_params,
            **suggested_peft_params,
        }
        trial_peft_params = PeftParams(**peft_params)

        trial_train_config = TrainingConfig(
            model=config.model,
            data=config.data,
            training=trial_training_params,
            peft=trial_peft_params,
        )

        # Build model for this trial
        use_peft = trial_training_params.use_peft

        model = build_model(
            model_params=config.model,
            peft_params=trial_peft_params if use_peft else None,
        )
        if use_peft:
            logger.info("Building PEFT model...")
            model = build_peft_model(
                model,
                trial_training_params.enable_gradient_checkpointing,
                trial_peft_params,
            )

        # Create trainer with suggested parameters
        create_trainer_fn = build_trainer(
            config.tuning.trainer_type, processor=processor, verbose=verbose
        )

        if is_local_process_zero():
            log_number_of_model_parameters(model)
            if trial_training_params.log_model_summary:
                log_model_summary(
                    model,
                    telemetry_dir / "model_summary.txt" if telemetry_dir else None,
                )

        collator: Optional[Callable] = build_collator_from_config(
            trial_train_config, tokenizer, debug=config.tuning.log_examples
        )

        # trl's SFTTrainer has its own dataset processing code. We should skip it if
        # the dataset is already processed, i.e. it's tokenized and has an `input_ids`
        # field. This generally occurs if the dataset is:
        # 1. In the Oumi registry and thus is processed by the `BasePretrainingDataset`
        # or `BaseSftDataset` classes
        # 2. Packing is requested, and thus is processed by the
        # `PretrainingAsyncTextDataset` class
        # See OPE-1108 for more details.
        if trial_training_params.trainer_type == TrainerType.TRL_SFT:
            example = next(iter(train_dataset))
            if "input_ids" in example:
                logger.info(
                    "Skipping dataset preparation for TRL_SFT trainer"
                    "since the dataset is already processed."
                )
                if "dataset_kwargs" not in trial_training_params.trainer_kwargs:
                    trial_training_params.trainer_kwargs["dataset_kwargs"] = {}
                # Skip preparing dataset if `skip_prepare_dataset` isn't already set.
                if (
                    "skip_prepare_dataset"
                    not in trial_training_params.trainer_kwargs["dataset_kwargs"]
                ):
                    trial_training_params.trainer_kwargs["dataset_kwargs"][
                        "skip_prepare_dataset"
                    ] = True

        _ensure_dir_exists(
            trial_training_params.output_dir, "trial_{trial_number}.train.output_dir"
        )
        device_cleanup()
        with torch_profile(
            trial_training_params.profiler,
            training_output_dir=trial_training_params.output_dir,
            record_function_name="oumi.tune",
        ) as profiler:
            with torch.profiler.record_function("create_trainer"):
                callbacks = build_training_callbacks(
                    trial_train_config,
                    model,
                    profiler,
                )

                trainer = create_trainer_fn(
                    model=model,
                    processing_class=tokenizer,
                    args=trial_training_params,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    callbacks=callbacks,
                    data_collator=collator,
                    # compute_metrics=metrics_functions,
                    **(trial_training_params.trainer_kwargs or {}),
                )
                assert isinstance(trainer, HuggingFaceTrainer)

                with torch.profiler.record_function("log_and_verify"):
                    log_nvidia_gpu_runtime_info(
                        log_prefix="GPU Metrics Before Training:"
                    )
                    verify_torch_distributed_initialized_if_needed()

                # Reclaim memory before training starts.
                device_cleanup()
                # Train
                trainer.train()

                # Save model and training config
                trainer.save_model(config=trial_train_config)
                trial_train_config.to_yaml(
                    Path(trial_training_params.output_dir, "training_config.yaml")
                )

                # Evaluate and extract metrics
                eval_results = trainer.get_last_eval_metrics()
                logger.info(f"Trial {trial_number} Evaluation results: {eval_results}")
        return {
            metric: eval_results[metric] for metric in config.tuning.evaluation_metrics
        }

    logger.info(f"Tuning init time: {time.time() - _START_TIME:.3f}s")
    logger.info("Starting hyperparameter tuning...")
    tuner.optimize(objective, n_trials=config.tuning.n_trials)

    if len(config.tuning.evaluation_direction) == 1:
        best_trial = tuner.get_best_trial()
        logger.info(f"Best trial: {best_trial}")
    else:
        best_trials = tuner.get_best_trials()
        logger.info(f"Best trials: {best_trials}")

    logger.info("Tuning is Complete. Saving study results...")
    tuner.save_study(config)

    log_nvidia_gpu_runtime_info(log_prefix="GPU Metrics After Tuning:")
    log_peak_gpu_memory()
    barrier()
    if is_distributed():
        cleanup_distributed()

    _log_feedback_request()
