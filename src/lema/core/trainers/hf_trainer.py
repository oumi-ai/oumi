from typing import Optional

import transformers

from lema.core.configs import TrainingConfig
from lema.core.distributed import is_world_process_zero
from lema.core.trainers.base_trainer import BaseTrainer
from lema.utils.logging import logger


class HuggingFaceTrainer(BaseTrainer):
    def __init__(self, hf_trainer: transformers.Trainer):
        """Initializes HuggingFace-specific Trainer version."""
        self._hf_trainer = hf_trainer

    def train(self, resume_from_checkpoint: Optional[str]) -> None:
        """Trains a model."""
        logger.info(
            f"Starting training with transformers=={transformers.__version__}..."
        )
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
        # TODO: OPE-213 - use safetensors to save model
        # Only save from "master" worker.
        if is_world_process_zero():
            # TODO: OPE-311 - Save full state dict for FSDP training.
            output_dir = config.training.output_dir

            self._hf_trainer.save_model(output_dir)

            logger.info(f"Model has been saved at {output_dir}.")
