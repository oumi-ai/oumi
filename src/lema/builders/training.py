from abc import ABC, abstractmethod
from typing import Callable, Optional, Type

import transformers
import trl

from lema.core.types import TrainerType, TrainingConfig
from lema.logging import logger


class BaseTrainer(ABC):
    @abstractmethod
    def train(self, resume_from_checkpoint: Optional[str]) -> None:
        """Trains a model."""
        pass

    @abstractmethod
    def save_state(self) -> None:
        """Saves state."""
        pass

    @abstractmethod
    def save_model(self, config: TrainingConfig) -> None:
        """Saves the model's state dictionary to the specified output directory.

        Args:
            config (TrainingConfig): The LeMa training config.

        Returns:
            None
        """
        pass


class HuggingFaceTrainer(BaseTrainer):
    def __init__(self, hf_trainer: transformers.Trainer):
        """Initializes HuggingFace-specific Trainer version."""
        self._hf_trainer = hf_trainer
        pass

    def train(self, resume_from_checkpoint: Optional[str]) -> None:
        """Trains a model."""
        self._hf_trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    def save_state(self) -> None:
        """Saves state."""
        self._hf_trainer.save_state()

    def save_model(self, config: TrainingConfig) -> None:
        """See base class."""
        output_dir = config.training.output_dir

        if config.training.use_peft:
            state_dict = {
                k: t
                for k, t in self._hf_trainer.model.named_parameters()
                if "lora_" in k
            }
        else:
            state_dict = self._hf_trainer.model.state_dict()

        self._hf_trainer._save(output_dir, state_dict=state_dict)
        logger.info(f"Model has been saved at {output_dir}.")


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

    def _create_builder_fn(
        cls: Type[transformers.Trainer],
    ) -> Callable[..., BaseTrainer]:
        return lambda *args, **kwargs: HuggingFaceTrainer(cls(*args, **kwargs))

    if trainer_type == TrainerType.TRL_SFT:
        return _create_builder_fn(trl.SFTTrainer)
    elif trainer_type == TrainerType.TRL_DPO:
        return _create_builder_fn(trl.DPOTrainer)
    elif trainer_type == TrainerType.HF:
        return _create_builder_fn(transformers.Trainer)

    raise NotImplementedError(f"Trainer type {trainer_type} not supported.")
