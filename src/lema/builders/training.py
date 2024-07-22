from typing import Callable, Type, cast

import transformers
import trl

from lema.core.trainers.hf_trainer import HuggingFaceTrainer
from lema.core.trainers.lema_trainer import Trainer as LemaTrainer
from lema.core.types import TrainerType, TrainingParams
from lema.core.types.base_trainer import BaseTrainer


def build_trainer(trainer_type: TrainerType) -> Callable[..., BaseTrainer]:
    """Builds a trainer creator functor based on the provided configuration.

    Args:
        trainer_type (TrainerType): Enum indicating the type of training.

    Returns:
        A builder function that can create an appropriate trainer based on the trainer
        type specified in the configuration. All function arguments supplied by caller
        are forwarded to the trainer's constructor.

    Raises:
        NotImplementedError: If the trainer type specified in the
            configuration is not supported.
    """

    def _create_hf_builder_fn(
        cls: Type[transformers.Trainer],
    ) -> Callable[..., BaseTrainer]:
        def _init_hf_trainer(*args, **kwargs) -> BaseTrainer:
            training_args = kwargs.pop("args", None)
            if training_args is not None:
                # if set, convert to HuggingFace Trainer args format
                training_args = cast(TrainingParams, training_args)
            return HuggingFaceTrainer(cls(*args, **kwargs, args=training_args.to_hf()))

        return _init_hf_trainer

    if trainer_type == TrainerType.TRL_SFT:
        return _create_hf_builder_fn(trl.SFTTrainer)
    elif trainer_type == TrainerType.TRL_DPO:
        return _create_hf_builder_fn(trl.DPOTrainer)
    elif trainer_type == TrainerType.HF:
        return _create_hf_builder_fn(transformers.Trainer)
    elif trainer_type == TrainerType.LEMA:
        return lambda *args, **kwargs: LemaTrainer(*args, **kwargs)

    raise NotImplementedError(f"Trainer type {trainer_type} not supported.")
