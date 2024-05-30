import argparse

from omegaconf import OmegaConf

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
    config_path, verbose, arg_list = parse_cli()

    limit_per_process_memory()
    device_cleanup()

    # Override with configuration file if provided
    if config_path is not None:
        config = TrainingConfig.from_yaml(config_path)
    else:
        config = OmegaConf.structured(TrainingConfig)

    # Override with CLI arguments if provided
    cli_config = OmegaConf.from_cli(arg_list)
    try:
        config = OmegaConf.merge(config, cli_config)
    except Exception as e:
        logger.exception(
            f"Failed to merge Omega config: {config} and CLI config: {cli_config}"
        )
        raise

    # Merge and validate configs
    config: TrainingConfig = OmegaConf.to_object(config)

    #
    # Run training
    #
    train(config)

    device_cleanup()


def train(config: TrainingConfig) -> None:
    """Train a model using the provided configuration."""
    log_devices_info()

    # Initialize model and tokenizer
    tokenizer = build_tokenizer(config.model)

    model = build_model(config)
    if config.training.use_peft:
        model = build_peft_model(
            model, config.training.enable_gradient_checkpointing, config.peft
        )

    if config.training.enable_gradient_checkpointing:
        model.enable_input_require_grads()

    # Load data & preprocessing
    dataset = build_dataset(dataset_config=config.data, tokenizer=tokenizer)

    # Train model
    trainer_cls = build_trainer(config.training.trainer_type)

    trainer = trainer_cls(
        model=model,
        tokenizer=tokenizer,
        args=config.training.to_hf(),
        train_dataset=dataset,
        **config.data.trainer_kwargs,
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
