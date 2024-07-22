from typing import Optional

from lema.core.types.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    def __init__(
        self,
        **kwargs,
    ):
        """Initializes the LeMa trainer."""
        raise NotImplementedError

    def train(self, resume_from_checkpoint: Optional[str] = None):
        """Trains the model."""
        raise NotImplementedError
