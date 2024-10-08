import argparse
import gc
import random
import time
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np
import torch
from transformers.trainer_utils import get_last_checkpoint

from oumi.builders import (
    build_data_collator,
    build_dataset_mixture,
    build_metrics_function,
    build_model,
    build_peft_model,
    build_tokenizer,
    build_trainer,
    build_training_callbacks,
)
from oumi.core.configs import (
    DatasetSplit,
    DatasetSplitParams,
    TrainerType,
    TrainingConfig,
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
    verify_torch_distributed_initialized_if_needed,
)
from oumi.core.trainers import BaseTrainer
from oumi.performance.torch_profiler_utils import torch_profile
from oumi.utils.device_utils import (
    log_nvidia_gpu_runtime_info,
)
from oumi.utils.io_utils import save_json
from oumi.utils.logging import configure_logger, logger
from oumi.utils.torch_utils import (
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
    """Main entry point for training Oumi.

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
    config.validate()

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


def _build_collator_if_needed(config: TrainingConfig, tokenizer) -> Optional[Any]:
    """Creates data collator if specified in config."""
    train_split: DatasetSplitParams = config.data.get_split(DatasetSplit.TRAIN)
    if not train_split.collator_name:
        return None

    return build_data_collator(
        collator_name=train_split.collator_name,
        tokenizer=tokenizer,
        max_length=config.model.model_max_length,
    )


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

    # FIXME OPE-229 Consider moving hardware capability validations
    # from TrainingConfig `__post_init__` to this function.
    return config


def train(config: TrainingConfig, **kwargs) -> None:
    """Trains a model using the provided configuration."""
    _START_TIME = time.time()

    if is_distributed():
        init_distributed(timeout_minutes=config.training.nccl_default_timeout_minutes)

    _create_training_dirs(config)
    _log_training_info(config)

    # Configure logging to file
    log_dir = Path(config.training.output_dir) / "logs"
    configure_logger("oumi", level=config.training.log_level, log_dir=log_dir)

    telemetry_dir = config.training.telemetry_dir

    config = _finalize_training_config(config)

    if is_local_process_zero():
        log_training_config(config)
        if telemetry_dir and is_world_process_zero():
            config.to_yaml(str(telemetry_dir / "training_config.yaml"))

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

    eval_dataset = None
    if len(config.data.get_split(DatasetSplit.VALIDATION).datasets) != 0:
        eval_dataset = build_dataset_mixture(config, tokenizer, DatasetSplit.VALIDATION)

    # Train model
    create_trainer_fn: Callable[..., BaseTrainer] = build_trainer(
        config.training.trainer_type
    )

    metrics_function = build_metrics_function(config.training)

    # Reclaim memory before training starts.
    gc.collect()

    collator = _build_collator_if_needed(config, tokenizer)

    with torch_profile(
        config.training.profiler,
        training_output_dir=config.training.output_dir,
        record_function_name="oumi.train",
    ) as profiler:
        with torch.profiler.record_function("create_trainer"):
            kwargs = {}
            if config.training.trainer_type == TrainerType.OUMI:
                kwargs["fsdp_params"] = config.fsdp

            callbacks = build_training_callbacks(config, model, profiler)

            trainer = create_trainer_fn(
                model=model,
                tokenizer=tokenizer,
                args=config.training,
                train_dataset=dataset,
                eval_dataset=eval_dataset,
                compute_metrics=metrics_function,
                callbacks=callbacks,
                data_collator=collator,
                **kwargs,
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

        with torch.profiler.record_function("wait_for_all_ranks"):
            # Make sure all workers start training at the same time.
            barrier()

        with torch.profiler.record_function("train"):
            logger.info(f"Training init time: {time.time() - _START_TIME}s")
            logger.info("Starting training...")
            trainer.train(resume_from_checkpoint=checkpoint_location)

    logger.info("Training is Complete.")

    log_nvidia_gpu_runtime_info(log_prefix="GPU Metrics After Training:")

    # Save final checkpoint & training state.
    if config.training.save_final_model:
        logger.info("Saving final state...")
        trainer.save_state()

        logger.info("Saving final model...")
        trainer.save_model(config=config)

    barrier()

    if is_distributed():
        cleanup_distributed()


if __name__ == "__main__":
    main()
