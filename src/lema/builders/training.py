from transformers import Trainer
from trl import DPOTrainer, SFTTrainer

from lema.core.types import TrainerType


def build_trainer(trainer_type: TrainerType):
    """Builds and returns a trainer based on the provided configuration.

    Args:
        trainer_type (TrainerType): Enum indicating the type of training.

    Returns:
        Trainer: An instance of the appropriate trainer based on the trainer type
            specified in the configuration.

    Raises:
        NotImplementedError: If the trainer type specified in the
            configuration is not supported.
    """
    if trainer_type == TrainerType.TRL_SFT:
        return SFTTrainer

    elif trainer_type == TrainerType.TRL_DPO:
        return DPOTrainer

    elif trainer_type == TrainerType.HF:
        return Trainer

    raise NotImplementedError(f"Trainer type {trainer_type} not supported.")
