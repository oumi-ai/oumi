from transformers import Trainer
from trl import DPOTrainer, SFTTrainer

from lema.core.types import TrainingConfig


def build_trainer(config: TrainingConfig):
    """Builds and returns a trainer based on the provided configuration.

    Args:
       config (TrainingConfig): The configuration object
       containing the training parameters.

    Returns:
       Trainer: An instance of the appropriate trainer based on the trainer type
       specified in the configuration.

    Raises:
       NotImplementedError: If the trainer type specified in the
       configuration is not supported.
    """

    if config.training_params.trainer_type == "trl_sft":
        return SFTTrainer

    elif config.training_params.trainer_type == "trl_dpo":
        return DPOTrainer

    elif config.training_params.trainer_type == "hf":
        return Trainer

    raise NotImplementedError(
        f"Trainer type {config.training_params.trainer_type} not supported."
    )
