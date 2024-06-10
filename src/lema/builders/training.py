from typing import Any, Callable, Dict, Optional, Type

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

    def _create_builder_fn(
        cls: Type[Trainer], extra_args: Dict[str, Any]
    ) -> Callable[..., Trainer]:
        return lambda *args, **kwargs: cls(*args, **{**kwargs, **extra_args})

    extra_args = {}
    if trainer_type == TrainerType.TRL_SFT:
        # if max_seq_length is not None:
        #    extra_args["max_seq_length"] = int(max_seq_length)
        return _create_builder_fn(SFTTrainer, extra_args)
    elif trainer_type == TrainerType.TRL_DPO:
        # if max_seq_length is not None:
        #    extra_args["max_length"] = int(max_seq_length)
        # DPOTrainer also defines "max_prompt_length" and "max_target_length".
        # How to handle that?
        return _create_builder_fn(DPOTrainer, extra_args)
    elif trainer_type == TrainerType.HF:
        return _create_builder_fn(Trainer, extra_args)

    raise NotImplementedError(f"Trainer type {trainer_type} not supported.")
