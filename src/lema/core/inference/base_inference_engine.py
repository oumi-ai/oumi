from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseInferenceEngine(ABC):
    """Base interface for running model inference."""

    @abstractmethod
    def infer(self, input: Any, output_filepath: Optional[str] = None, **kwargs) -> Any:
        """Runs model inference.

        Args:
            input: Input data to run inference on.
            output_filepath: Path to the file where the output should be written.
            **kwargs: Additional arguments used for inference.

        Returns:
            Any: Inference output.
        """
        raise NotImplementedError
