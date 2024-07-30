import argparse
import pathlib
import random
import time
from typing import Callable, Optional

import numpy as np
import torch
from transformers.trainer_utils import get_last_checkpoint

from lema.builders import (
    build_dataset,
    build_metrics_function,
    build_model,
    build_peft_model,
    build_tokenizer,
    build_trainer,
)
from lema.core.callbacks.mfu_callback import MfuTrainerCallback
from lema.core.distributed import (
    cleanup_distributed,
    get_device_rank_info,
    init_distributed,
    is_distributed,
    is_local_process_zero,
    is_world_process_zero,
    verify_torch_distributed_initialized_if_needed,
)
from lema.core.types import DatasetSplit, TrainingConfig
from lema.core.types.base_trainer import BaseTrainer
from lema.utils.debugging_utils import log_nvidia_gpu_memory_utilization
from lema.utils.logging import configure_logger, logger
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

_START_TIME = -1.0


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

    config: TrainingConfig = TrainingConfig.from_yaml_and_arg_list(
        config_path, arg_list, logger=logger
    )

    limit_per_process_memory()
    device_cleanup()
    set_random_seeds(config.training.seed)

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


def set_random_seeds(seed: int = 42, set_deterministic: bool = False) -> None:
    """Set random seeds for reproducibility.

    Each worker will have a different seed to ensure that each worker
    starts with a different random state.

    Args:
        seed: The seed value to set for random number generators.
        set_deterministic: Whether to set deterministic mode for CUDA operations.
    """
    device_info = get_device_rank_info()

    local_seed = seed + device_info.rank

    logger.info(f"Setting random seed to {local_seed} on rank {device_info.rank}.")
    random.seed(local_seed)
    np.random.seed(local_seed)
    torch.manual_seed(local_seed)
    torch.cuda.manual_seed(local_seed)

    if set_deterministic:
        logger.info("Setting deterministic mode for CUDA operations.")
        torch.backends.cudnn.deterministic = True


def train(config: TrainingConfig, **kwargs) -> None:
    """Trains a model using the provided configuration."""
    _START_TIME = time.time()

    if is_distributed():
        init_distributed()

    if is_local_process_zero():
        log_versioning_info()
        log_devices_info()
        log_training_config(config)
        _ensure_training_output_dir_exists(config.training.output_dir)

    # Configure logging to file
    log_dir = pathlib.Path(config.training.output_dir) / "logs"
    configure_logger("lema", level=config.training.log_level, log_dir=log_dir)

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
        if is_local_process_zero():
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

    metrics_function = build_metrics_function(config.training)

    training_callbacks = []
    if config.training.include_performance_metrics:
        if config.model.model_max_length is None:
            raise ValueError(
                "model_max_length must be set to log performance information."
            )
        if not torch.cuda.is_available():
            logger.warning("MFU logging is only supported on GPU. Skipping callback.")
        else:
            num_total_params = count_model_parameters(model)
            num_mfu_params = (
                num_total_params.all_params - num_total_params.embedding_params
            )
            logger.info(f"Number of model parameters for MFU: {num_mfu_params:,}")
            # Ignore attention and rematerialization to ensure metric matches most
            # common implementations.
            mfu_callback = MfuTrainerCallback(
                dtype=model.dtype,
                num_params=num_mfu_params,
                sequence_length=config.model.model_max_length,
            )
            training_callbacks.append(mfu_callback)

    trainer = create_trainer_fn(
        model=model,
        tokenizer=tokenizer,
        args=config.training,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        compute_metrics=metrics_function,
        callbacks=training_callbacks,
        **config.training.trainer_kwargs,
    )

    logger.info("Max Memory Usage Before Training: ")
    log_nvidia_gpu_memory_utilization()

    logger.info(f"Training init time: {time.time() - _START_TIME}s")
    logger.info("Starting training...")
    with torch_profile(
        config.training.profiler,
        training_output_dir=config.training.output_dir,
        record_function_name="lema.train",
    ):
        verify_torch_distributed_initialized_if_needed()
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
    if is_world_process_zero():
        trainer.save_state()
        if config.training.save_final_model:
            trainer.save_model(config=config)

    if is_distributed():
        cleanup_distributed()


if __name__ == "__main__":
    main()
