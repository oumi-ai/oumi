from abc import ABC, abstractmethod
from typing import List, Optional

from lema.core.configs import GenerationConfig
from lema.core.types.turn import Conversation
from lema.utils.logging import logger


class BaseInferenceEngine(ABC):
    """Base class for running model inference."""

    def infer(
        self,
        input: Optional[List[Conversation]] = None,
        generation_config: Optional[GenerationConfig] = None,
    ) -> Optional[List[Conversation]]:
        """Runs model inference.

        Args:
            input: A list of conversations to run inference on. Optional.
            generation_config: Configuration parameters for generation during inference.
                If not specified, a default config is inferred.

        Returns:
            Optional[List[Conversation]]: Inference output. Returns None if the output
                is written to a file.
        """
        if (
            input is not None
            and generation_config is not None
            and generation_config.input_filepath is not None
        ):
            raise ValueError(
                "Only one of input or generation_config.input_filepath should be "
                "provided."
            )
        if generation_config is None:
            logger.warning("No generation config provided. Using the default config.")
            generation_config = GenerationConfig()
        if input is not None:
            return self.infer_online(input, generation_config)
        elif generation_config.input_filepath is not None:
            return self.infer_from_file(
                generation_config.input_filepath, generation_config
            )
        else:
            raise ValueError(
                "One of input or generation_config.input_filepath must be provided."
            )

    @abstractmethod
    def infer_online(
        self, input: List[Conversation], generation_config: GenerationConfig
    ) -> Optional[List[Conversation]]:
        """Runs model inference online.

        Args:
            input: A list of conversations to run inference on.
            generation_config: Configuration parameters for generation during inference.

        Returns:
            Optional[List[Conversation]]: Inference output. Returns None if the output
                is written to a file.
        """
        raise NotImplementedError

    @abstractmethod
    def infer_from_file(
        self, input_filepath: str, generation_config: GenerationConfig
    ) -> Optional[List[Conversation]]:
        """Runs model inference on inputs in the provided file.

        This is a convenience method to prevent boilerplate from asserting the existence
        of input_filepath in the generation_config.

        Args:
            input_filepath: Path to the input file containing prompts for generation.
            generation_config: Configuration parameters for generation during inference.

        Returns:
            Optional[List[Conversation]]: Inference output. Returns None if the output
                is written to a file.
        """
        raise NotImplementedError
