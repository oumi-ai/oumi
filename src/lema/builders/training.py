from transformers import Trainer
from trl import DPOTrainer, SFTTrainer

from lema.core.types import TrainerType, TrainingConfig


def build_trainer(config: TrainingConfig):
    """Build and returns a trainer based on the provided configuration.

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
    if config.training.trainer_type == TrainerType.TRL_SFT:
        return SFTTrainer

    elif config.training.trainer_type == TrainerType.TRL_DPO:
        return DPOTrainer

    elif config.training.trainer_type == TrainerType.HF:
        return Trainer

    raise NotImplementedError(
        f"Trainer type {config.training.trainer_type} not supported."
    )
