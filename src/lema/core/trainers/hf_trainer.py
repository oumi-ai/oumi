from typing import Optional

import transformers

from lema.core.distributed import is_world_process_zero
from lema.core.types import TrainingConfig
from lema.core.types.base_trainer import BaseTrainer
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
        # TODO: OPE-213 use safetensors to save model
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
                self._hf_trainer._save(output_dir, state_dict=state_dict)
            else:
                self._hf_trainer.save_model(output_dir)

            logger.info(f"Model has been saved at {output_dir}.")
