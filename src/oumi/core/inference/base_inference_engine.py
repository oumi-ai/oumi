from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Set

import jsonlines

from oumi.core.configs import GenerationParams
from oumi.core.types.turn import Conversation
from oumi.utils.logging import logger


class BaseInferenceEngine(ABC):
    """Base class for running model inference."""

    def infer(
        self,
        input: Optional[List[Conversation]] = None,
        generation_params: Optional[GenerationParams] = None,
    ) -> List[Conversation]:
        """Runs model inference.

        Args:
            input: A list of conversations to run inference on. Optional.
            generation_params: Parameters for generation during inference.
                If not specified, a default config is inferred.

        Returns:
            List[Conversation]: Inference output.
        """
        if (
            input is not None
            and generation_params is not None
            and generation_params.input_filepath is not None
        ):
            raise ValueError(
                "Only one of input or generation_params.input_filepath should be "
                "provided."
            )
        if generation_params is None:
            logger.warning("No generation params provided. Using the default params.")
            generation_params = GenerationParams()

        self._check_unsupported_params(generation_params)

        if input is not None:
            return self.infer_online(input, generation_params)
        elif generation_params.input_filepath is not None:
            return self.infer_from_file(
                generation_params.input_filepath, generation_params
            )
        else:
            raise ValueError(
                "One of input or generation_params.input_filepath must be provided."
            )

    def _read_conversations(self, input_filepath: str) -> List[Conversation]:
        """Reads conversations from a file in Oumi chat format.

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

    def _get_scratch_filepath(self, output_filepath: str) -> str:
        """Returns a scratch filepath for the given output filepath.

        For example, if the output filepath is "/foo/bar/output.json", the scratch
        filepath will be "/foo/bar/scratch/output.json"

        Args:
            output_filepath: The output filepath.

        Returns:
            str: The scratch filepath.
        """
        original_filepath = Path(output_filepath)
        return str(original_filepath.parent / "scratch" / original_filepath.name)

    def _save_conversation(
        self, conversation: Conversation, output_filepath: str
    ) -> None:
        """Appends a conversation to a file in Oumi chat format.

        Args:
            conversation: The conversation to save.
            output_filepath: The path to the file where the conversation should be
                saved.
        """
        # Make the directory if it doesn't exist.
        Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)
        with jsonlines.open(output_filepath, mode="a") as writer:
            json_obj = conversation.model_dump()
            writer.write(json_obj)

    def _save_conversations(
        self, conversations: List[Conversation], output_filepath: str
    ) -> None:
        """Saves conversations to a file in Oumi chat format.

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

    def _check_unsupported_params(self, generation_params: GenerationParams):
        """Checks for unsupported parameters and logs warnings."""
        supported_params = self.get_supported_params()
        all_params = set(generation_params.__dict__.keys())
        unsupported_params = all_params - supported_params

        for param in unsupported_params:
            value = getattr(generation_params, param)
            if value is not None and value != 0 and value != {} and value != []:
                logger.warning(
                    f"{self.__class__.__name__} does not support {param}. "
                    f"Received value: {param}={value}. This parameter will be ignored."
                )

    @abstractmethod
    def get_supported_params(self) -> Set[str]:
        """Returns a set of supported generation parameters for this engine.

        Override this method in derived classes to specify which parameters
        are supported.

        Returns:
            Set[str]: A set of supported parameter names.
        """
        raise NotImplementedError

    @abstractmethod
    def infer_online(
        self, input: List[Conversation], generation_params: GenerationParams
    ) -> List[Conversation]:
        """Runs model inference online.

        Args:
            input: A list of conversations to run inference on.
            generation_params: Parameters for generation during inference.

        Returns:
            List[Conversation]: Inference output.
        """
        raise NotImplementedError

    @abstractmethod
    def infer_from_file(
        self, input_filepath: str, generation_params: GenerationParams
    ) -> List[Conversation]:
        """Runs model inference on inputs in the provided file.

        This is a convenience method to prevent boilerplate from asserting the existence
        of input_filepath in the generation_params.

        Args:
            input_filepath: Path to the input file containing prompts for generation.
            generation_params: Parameters for generation during inference.

        Returns:
            List[Conversation]: Inference output.
        """
        raise NotImplementedError

    def apply_chat_template(
        self, conversation: Conversation, **tokenizer_kwargs
    ) -> str:
        """Applies the chat template to the conversation.

        Args:
            conversation: The conversation to apply the chat template to.
            tokenizer_kwargs: Additional keyword arguments to pass to the tokenizer.

        Returns:
            str: The conversation with the chat template applied.
        """
        tokenizer = getattr(self, "_tokenizer", None)

        if tokenizer is None:
            raise ValueError("Tokenizer is not initialized.")

        if tokenizer.chat_template is None:
            raise ValueError("Tokenizer does not have a chat template.")

        if "tokenize" not in tokenizer_kwargs:
            tokenizer_kwargs["tokenize"] = False

        return tokenizer.apply_chat_template(conversation, **tokenizer_kwargs)
