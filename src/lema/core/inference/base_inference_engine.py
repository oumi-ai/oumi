from abc import ABC, abstractmethod
from typing import List, Optional

from lema.core.types.turn import Conversation


class BaseInferenceEngine(ABC):
    """Base class for running model inference."""

    def infer(
        self, input: List[Conversation], output_filepath: Optional[str] = None, **kwargs
    ) -> Optional[List[Conversation]]:
        """Runs model inference.

        Args:
            input: A list of conversations to run inference on.
            output_filepath: Path to the file where the output should be written.
                Optional.
            **kwargs: Additional arguments used for inference.

        Returns:
            Optional[List[Conversation]]: Inference output. Returns None if
                output_filepath is provided.
        """
        if output_filepath is not None:
            return self.infer_batch(input, output_filepath, **kwargs)
        return self.infer_online(input, **kwargs)

    @abstractmethod
    def infer_online(self, input: List[Conversation], **kwargs) -> List[Conversation]:
        """Runs model inference online.

        Args:
            input: A list of conversations to run inference on.
            **kwargs: Additional arguments used for inference.

        Returns:
            Optional[List[Conversation]]: Inference output.
        """
        raise NotImplementedError

    @abstractmethod
    def infer_batch(self, input: List[Conversation], output_filepath: str, **kwargs):
        """Runs model inference in batch mode.

        Args:
            input: A list of conversations to run inference on.
            output_filepath: Path to the file where the output should be written.
            **kwargs: Additional arguments used for inference.
        """
        raise NotImplementedError
