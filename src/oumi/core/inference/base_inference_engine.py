import queue
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

import jsonlines

from oumi.core.configs import GenerationConfig
from oumi.core.types.turn import Conversation
from oumi.utils.logging import logger


class BaseInferenceEngine(ABC):
    """Base class for running model inference."""

    def __init__(self):
        """Initializes the BaseInferenceEngine.

        Sets up a queue and a background thread for writing conversations to files.
        """
        self._write_queue = queue.Queue()

        def _write_conversation_thread():
            while True:
                conversation, output_filepath = self._write_queue.get()
                # Make the directory if it doesn't exist.
                Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)
                with jsonlines.open(output_filepath, mode="a") as writer:
                    json_obj = conversation.model_dump()
                    writer.write(json_obj)
                self._write_queue.task_done()

        threading.Thread(target=_write_conversation_thread, daemon=True).start()

    def __del__(self):
        """Closes the write queue before being deleted."""
        self._write_queue.join()

    def infer(
        self,
        input: Optional[List[Conversation]] = None,
        generation_config: Optional[GenerationConfig] = None,
    ) -> List[Conversation]:
        """Runs model inference.

        Args:
            input: A list of conversations to run inference on. Optional.
            generation_config: Configuration parameters for generation during inference.
                If not specified, a default config is inferred.

        Returns:
            List[Conversation]: Inference output.
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

    def _save_conversation(
        self, conversation: Conversation, output_filepath: str
    ) -> None:
        """Saves single conversation to a file in Oumi chat format.

        Args:
            conversation: A single conversation to save.
            output_filepath: The filepath to where the conversation should be saved.
        """
        print("save")
        self._write_queue.put((conversation, output_filepath))

    def _finish_writing(self):
        """Blocks until all conversations are written to file."""
        self._write_queue.join()

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
