import transformers

from lema.core.types import TrainingConfig


def save_model(config: TrainingConfig, trainer: transformers.Trainer) -> None:
    """Save the model's state dictionary to the specified output directory.

    Args:
        config (TrainingConfig): The LeMa training config.
        trainer (transformers.Trainer): The trainer object used for training the model.

    Returns:
        None
    """
    output_dir = config.training.output_dir

    if trainer.model.use_peft:
        state_dict = {k: t for k, t in trainer.model.named_parameters() if "lora_" in k}
    else:
        state_dict = trainer.model.state_dict()

    trainer._save(output_dir, state_dict=state_dict)
