import argparse
from typing import Callable, Optional

from transformers import Trainer
from transformers.trainer_utils import get_last_checkpoint

from lema.builders import (
    build_dataset,
    build_model,
    build_peft_model,
    build_tokenizer,
    build_trainer,
)
from lema.core.types import TrainingConfig
from lema.logging import logger
from lema.utils.saver import save_model
from lema.utils.torch_utils import (
    device_cleanup,
    limit_per_process_memory,
    log_devices_info,
    log_model_summary,
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


def train(config: TrainingConfig, **kwargs) -> None:
    """Trains a model using the provided configuration."""
    log_versioning_info()
    log_devices_info()

    # Initialize model and tokenizer
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

    # Enable gradients for input embeddings
    if config.training.enable_gradient_checkpointing:
        model.enable_input_require_grads()

    # Load data & preprocessing
    dataset = build_dataset(config, tokenizer)

    # Train model
    create_trainer_fn: Callable[..., Trainer] = build_trainer(
        config.training.trainer_type
    )

    trainer = create_trainer_fn(
        model=model,
        tokenizer=tokenizer,
        args=config.training.to_hf(),
        train_dataset=dataset,
        **config.data.trainer_kwargs,
    )

    logger.info("Starting training...")
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

    # Save final checkpoint & training state
    # FIXME: add conditional saving logic for multi-node runs.
    trainer.save_state()

    save_model(
        config=config,
        trainer=trainer,
    )


if __name__ == "__main__":
    main()
