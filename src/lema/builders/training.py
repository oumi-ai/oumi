from typing import Callable, Optional, Type

import torch.distributed
import transformers
import trl

from lema.core.types import TrainerType, TrainingConfig
from lema.core.types.base_trainer import BaseTrainer
from lema.logging import logger
from lema.utils.torch_utils import is_world_process_zero


class HuggingFaceTrainer(BaseTrainer):
    def __init__(self, hf_trainer: transformers.Trainer):
        """Initializes HuggingFace-specific Trainer version."""
        self._hf_trainer = hf_trainer

    def train(self, resume_from_checkpoint: Optional[str]) -> None:
        """Trains a model."""
        self._hf_trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    def save_state(self) -> None:
        """See base class.

        Saves the Trainer state, since Trainer.save_model saves only the tokenizer
        with the model.

        HuggingFace normally writes state into "trainer_state.json" under output_dir.
        """
        self._hf_trainer.save_state()

    def save_model(self, config: TrainingConfig) -> None:
        """See base class."""
        if is_world_process_zero():
            # Only save from "master" worker.
            output_dir = config.training.output_dir

            if config.training.use_peft:
                state_dict = {
                    k: t
                    for k, t in self._hf_trainer.model.named_parameters()
                    if "lora_" in k
                }
                # FIXME: Can we replace the private method `_save()` with
                # `Trainer.save_model()`?
                # https://github.com/huggingface/transformers/blob/0f67ba1d741d65b07d549daf4ee157609ce4f9c1/src/transformers/trainer.py#L3384
                # FIXME: Add conditional saving logic for multi-node runs.
                self._hf_trainer._save(output_dir, state_dict=state_dict)
            else:
                self._hf_trainer.save_model(output_dir)

            logger.info(f"Model has been saved at {output_dir}.")

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # Make sure all workers are waiting until saving is done
            # so they continue to other tasks in unison.
            torch.distributed.barrier()


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
