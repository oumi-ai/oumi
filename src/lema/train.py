import argparse
from typing import Callable

from transformers import Trainer

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
)


def parse_cli():
    """Parse command line arguments and return the configuration filename."""
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

    #
    # Run training
    #
    train(config)

    device_cleanup()


def train(config: TrainingConfig, **kwargs) -> None:
    """Train a model using the provided configuration."""
    log_devices_info()

    # Initialize model and tokenizer
    tokenizer = build_tokenizer(config.model)

    model = build_model(config, *kwargs)
    if config.training.use_peft:
        model = build_peft_model(
            model, config.training.enable_gradient_checkpointing, config.peft
        )

    # Enable gradients for input embeddings
    if config.training.enable_gradient_checkpointing:
        model.enable_input_require_grads()

    # Load data & preprocessing
    dataset = build_dataset(config, tokenizer)

    # Train model
    create_trainer_fn: Callable[..., Trainer] = build_trainer(
        config.training.trainer_type, config.model.model_max_length
    )

    trainer = create_trainer_fn(
        model=model,
        tokenizer=tokenizer,
        args=config.training.to_hf(),
        train_dataset=dataset,
        **kwargs,
    )

    logger.info("Starting training...")
    trainer.train()
    logger.info("Training is Complete.")

    # Save final checkpoint & training state
    trainer.save_state()

    save_model(
        config=config,
        trainer=trainer,
    )


if __name__ == "__main__":
    main()
