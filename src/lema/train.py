import transformers

from lema.builders import (
    build_dataset,
    build_model,
    build_peft_model,
    build_tokenizer,
    build_trainer,
)
from lema.core.saver import save_model
from lema.core.types import (
    DataParams,
    ModelParams,
    PeftParams,
    TrainingConfig,
    TrainingParams,
)


def main():
    """Main entry point for training LeMa."""

    #
    # Parse CLI arguments
    #
    parser = transformers.HfArgumentParser(
        (DataParams, ModelParams, TrainingParams, PeftParams)
    )

    data_params, model_params, training_params, peft_params = (
        parser.parse_args_into_dataclasses()
    )

    config = TrainingConfig(data_params, model_params, training_params, peft_params)

    #
    # Run training
    #
    train(config)


def train(config: TrainingConfig):
    """Train a model using the provided configuration."""

    # Initialize model and tokenizer
    tokenizer = build_tokenizer(config)

    model = build_model(config)

    if config.training_params.use_lora:
        model = build_peft_model(config)

    if config.training_params.gradient_checkpointing:
        model.enable_input_require_grads()

    # Load data & preprocessing
    dataset = build_dataset(tokenizer)

    trainer_cls = build_trainer(config)

    # Train model
    trainer = trainer_cls(
        model=model,
        tokenizer=tokenizer,
        args=config.training_params,
        train_dataset=dataset,
        **config.data_params.trainer_kwargs,
    )

    trainer.train()

    # Save final checkpoint & training state
    trainer.save_state()

    save_model(
        config=config,
        trainer=trainer,
    )


if __name__ == "__main__":
    main()
