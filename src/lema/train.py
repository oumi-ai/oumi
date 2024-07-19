import argparse
import pathlib
import time
from typing import Callable, Optional

import torch
from transformers.trainer_utils import get_last_checkpoint

from lema.builders import (
    build_dataset,
    build_model,
    build_peft_model,
    build_tokenizer,
    build_trainer,
)
from lema.core.callbacks.mfu_callback import MfuTrainerCallback
from lema.core.registry import REGISTRY
from lema.core.types import DatasetSplit, TrainingConfig
from lema.core.types.base_trainer import BaseTrainer
from lema.utils.debugging_utils import log_nvidia_gpu_memory_utilization
from lema.utils.logging import logger
from lema.utils.torch_profiler_utils import torch_profile
from lema.utils.torch_utils import (
    count_model_parameters,
    device_cleanup,
    limit_per_process_memory,
    log_devices_info,
    log_model_summary,
    log_training_config,
    log_versioning_info,
)


def parse_cli():
    """Parses command line arguments and returns the configuration filename."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", default=None, help="Path to the configuration file"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
    )
    args, unknown = parser.parse_known_args()
    return args.config, args.verbose, unknown


def main() -> None:
    """Main entry point for training LeMa.

    Training arguments are fetched from the following sources, ordered by
    decreasing priority:
    1. [Optional] Arguments provided as CLI arguments, in dotfile format
    2. [Optional] Arguments provided in a yaml config file
    3. Default arguments values defined in the data class
    """
    # Load configuration
    config_path, _verbose, arg_list = parse_cli()  # TODO: keep or not unused var

    limit_per_process_memory()
    device_cleanup()

    config: TrainingConfig = TrainingConfig.from_yaml_and_arg_list(
        config_path, arg_list, logger=logger
    )

    # Run training
    train(config)

    device_cleanup()


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


def _ensure_training_output_dir_exists(output_dir: str) -> None:
    if not output_dir:
        raise ValueError("training.output_dir is not specified!")
    output_dir_path: pathlib.Path = pathlib.Path(output_dir)
    if output_dir_path.exists():
        if not output_dir_path.is_dir():
            raise ValueError(f"training.output_dir='{output_dir}' is not a directory!")
    else:
        logger.info(f"Creating output dir: {output_dir}...")
        output_dir_path.mkdir(parents=True, exist_ok=True)
    logger.info(
        f"Training output dir absolute path : {str(output_dir_path.absolute())}"
    )


def train(config: TrainingConfig, **kwargs) -> None:
    """Trains a model using the provided configuration."""
    log_versioning_info()
    log_devices_info()
    log_training_config(config)
    start_time = time.time()

    _ensure_training_output_dir_exists(config.training.output_dir)

    # Initialize model and tokenizer.
    tokenizer = build_tokenizer(config.model)

    # Are we supporting PEFT?
    use_peft = config.training.use_peft and config.peft

    # Build model.
    model = build_model(
        model_params=config.model,
        peft_params=config.peft if use_peft else None,
        *kwargs,
    )

    if use_peft:
        model = build_peft_model(
            model, config.training.enable_gradient_checkpointing, config.peft
        )

    if config.training.log_model_summary:
        log_model_summary(model)

    # Enable gradient checkpointing
    if config.training.enable_gradient_checkpointing:
        model.gradient_checkpointing_enable(
            config.training.gradient_checkpointing_kwargs
        )

    # Load data & preprocessing
    dataset = build_dataset(config, tokenizer, DatasetSplit.TRAIN)

    eval_dataset = None
    if len(config.data.get_split(DatasetSplit.VALIDATION).datasets) != 0:
        eval_dataset = build_dataset(config, tokenizer, DatasetSplit.VALIDATION)

    # Train model
    create_trainer_fn: Callable[..., BaseTrainer] = build_trainer(
        config.training.trainer_type
    )

    metrics_function = None
    if config.training.metrics_function:
        metrics_function = REGISTRY.get_metrics_function(
            config.training.metrics_function
        )
        if not metrics_function:
            raise KeyError(
                f"metrics_function `{config.training.metrics_function}` "
                "was not found in the registry."
            )

    training_callbacks = []
    if config.training.include_performance_metrics:
        if config.model.model_max_length is None:
            raise ValueError(
                "model_max_length must be set to log performance information."
            )
        if not torch.cuda.is_available():
            logger.warning("MFU logging is only supported on GPU. Skipping callback.")
        else:
            num_params = count_model_parameters(model).all_params
            logger.info(f"Number of model parameters: {num_params}")
            mfu_callback = MfuTrainerCallback(
                dtype=model.dtype,
                num_params=num_params,
                start_time_seconds=start_time,
                sequence_length=config.model.model_max_length,
            )
            training_callbacks.append(mfu_callback)

    trainer = create_trainer_fn(
        model=model,
        tokenizer=tokenizer,
        args=config.training.to_hf(),
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        compute_metrics=metrics_function,
        callbacks=training_callbacks,
        **config.training.trainer_kwargs,
    )

    logger.info("Max Memory Usage Before Training: ")
    log_nvidia_gpu_memory_utilization()

    logger.info("Starting training...")
    with torch_profile(
        config.training.profiler,
        training_output_dir=config.training.output_dir,
        record_function_name="lema.train",
    ):
        trainer.train(
            resume_from_checkpoint=(
                _find_checkpoint_to_resume_from(
                    config.training.resume_from_checkpoint,
                    config.training.try_resume_from_last_checkpoint,
                    config.training.output_dir,
                )
            )
        )
    logger.info("Training is Complete.")

    logger.info("Max Memory Usage Before Training: ")
    log_nvidia_gpu_memory_utilization()

    # Save final checkpoint & training state.
    trainer.save_state()
    if config.training.save_final_model:
        trainer.save_model(config=config)


if __name__ == "__main__":
    main()
