from typing import Optional

from lema.core.types.base_trainer import BaseTrainer
from lema.core.types.configs import TrainingConfig


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

    def save_state(self) -> None:
        """Saves the Trainer state.

        Under distributed environment this is done only for a process with rank 0.
        """
        raise NotImplementedError

    def save_model(self, config: TrainingConfig) -> None:
        """Saves the model's state dictionary to the specified output directory.

        Args:
            config (TrainingConfig): The LeMa training config.

        Returns:
            None
        """
        raise NotImplementedError
