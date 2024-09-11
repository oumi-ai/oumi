from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

import jsonlines

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

    def _read_conversations(self, input_filepath: str) -> List[Conversation]:
        """Reads conversations from a file in LeMa chat format.

        Args:
            input_filepath: The path to the file containing the conversations.

        Returns:
            List[Conversation]: A list of conversations read from the file.
        """
        conversations = []
        with open(input_filepath) as f:
            for line in f:
                # Only parse non-empty lines.
                if line.strip():
                    conversation = Conversation.model_validate_json(line)
                    conversations.append(conversation)
        return conversations

    def _save_conversations(
        self, conversations: List[Conversation], output_filepath: str
    ) -> None:
        """Saves conversations to a file in LeMa chat format.

        Args:
            conversations: A list of conversations to save.
            output_filepath: The path to the file where the conversations should be
                saved.
        """
        # Make the directory if it doesn't exist.
        Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)
        with jsonlines.open(output_filepath, mode="w") as writer:
            for conversation in conversations:
                json_obj = conversation.model_dump()
                writer.write(json_obj)

    @abstractmethod
    def infer_online(
        self, input: List[Conversation], generation_config: GenerationConfig
    ) -> List[Conversation]:
        """Runs model inference online.

        Args:
            input: A list of conversations to run inference on.
            generation_config: Configuration parameters for generation during inference.

        Returns:
            List[Conversation]: Inference output.
        """
        raise NotImplementedError

    @abstractmethod
    def infer_from_file(
        self, input_filepath: str, generation_config: GenerationConfig
    ) -> List[Conversation]:
        """Runs model inference on inputs in the provided file.

        This is a convenience method to prevent boilerplate from asserting the existence
        of input_filepath in the generation_config.

        Args:
            input_filepath: Path to the input file containing prompts for generation.
            generation_config: Configuration parameters for generation during inference.

        Returns:
            List[Conversation]: Inference output.
        """
        raise NotImplementedError
