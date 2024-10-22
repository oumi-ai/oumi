import abc
from typing import List, Optional

import PIL.Image
import transformers

from oumi.core.processors.base_image_processor import (
    BaseImageProcessor,
)
from oumi.core.tokenizers.base_tokenizer import BaseTokenizer
from oumi.core.types.conversation import Message


class BaseProcessor(abc.ABC):
    """Base class for oumi processors.

    The high-level purpose of a processor is to generate model-specific input features
    from input data such as text, images, conversations, etc.
    """

    @property
    @abc.abstractmethod
    def tokenizer(self) -> BaseTokenizer:
        """Returns a tokenizer associated with this processor."""
        raise NotImplementedError

    @tokenizer.setter
    @abc.abstractmethod
    def tokenizer(self, new_tokenizer: BaseTokenizer) -> None:
        """Sets a tokenizer associated with this processor."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def chat_template(self) -> str:
        """Returns a chat template."""
        raise NotImplementedError

    @chat_template.setter
    @abc.abstractmethod
    def chat_template(self, new_chat_template: str) -> None:
        """Sets a tokenizer associated with this processor."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def image_processor(self) -> Optional[BaseImageProcessor]:
        """Returns an image processor."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def image_token(self) -> Optional[str]:
        """Returns an image token."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def image_token_id(self) -> Optional[int]:
        """Returns an image token id."""
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(
        self,
        *,
        text: List[str],
        padding: bool,
        images: Optional[List[PIL.Image.Image]] = None,
        return_tensors: Optional[str] = "pt",
    ) -> transformers.BatchEncoding:
        """Invokes the processor to extract features.

        Args:
            text: A list of text prompts.
            padding: Whether to pad sequences to common length.
            images: A list of input images.
            return_tensors: The format of returned tensors.

        Returns:
            transformers.BatchEncoding: The model-specific input features.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def apply_chat_template(
        self, conversation: List[Message], add_generation_prompt: bool = False
    ) -> str:
        """Applies a chat template.

        Args:
            conversation: A list of messages (conversation "turns").
            add_generation_prompt: Whether to append generation prompt to the output.

        Returns:
            A text prompt, which includes all input messages formatted into a string.
        """
        raise NotImplementedError
