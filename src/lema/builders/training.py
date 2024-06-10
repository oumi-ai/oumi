from typing import Callable, Optional, Type

from transformers import Trainer
from trl import DPOTrainer, SFTTrainer

from lema.core.types import TrainerType


def build_trainer(
    trainer_type: TrainerType, max_seq_length: Optional[int]
) -> Callable[..., Trainer]:
    """Builds a trainer creator functor based on the provided configuration.

    Args:
        trainer_type (TrainerType): Enum indicating the type of training.
        max_seq_length: Maximum sequence length (tokens).

    Returns:
        A builder function that can create an appropriate trainer based on the trainer
        type specified in the configuration. All function arguments supplied by caller
        are forwarded to the trainer's constructor.

    Raises:
        NotImplementedError: If the trainer type specified in the
            configuration is not supported.
    """

    def _create_builder_fn(cls: Type[Trainer]) -> Callable[..., Trainer]:
        return lambda *args, **kwargs: cls(*args, **kwargs)

    if trainer_type == TrainerType.TRL_SFT:
        return _create_builder_fn(SFTTrainer)
    elif trainer_type == TrainerType.TRL_DPO:
        return _create_builder_fn(DPOTrainer)
    elif trainer_type == TrainerType.HF:
        return _create_builder_fn(Trainer)

    raise NotImplementedError(f"Trainer type {trainer_type} not supported.")
