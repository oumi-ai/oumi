import abc
from typing import Any, Callable, Dict, List, Optional, Union

import PIL.Image
from typing_extensions import override

from oumi.core.processors.base_image_processor import (
    BaseImageProcessor,
    DefaultImageProcessor,
)
from oumi.core.tokenizers.base_tokenizer import BaseTokenizer


class BaseProcessor(abc.ABC):
    """Base class for oumi image processors."""

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
        text: Union[str, List[str]],
        images: Union[PIL.Image.Image, List[PIL.Image.Image]],
        padding: bool,
        return_tensors: str = "pt",
    ) -> Dict[str, Any]:
        """Invokes the processor to extract features."""
        raise NotImplementedError

    @abc.abstractmethod
    def apply_chat_template(
        self, texts: Union[str, List[str]], add_generation_prompt: bool = False
    ) -> str:
        """Applies chat template."""
        raise NotImplementedError


class DefaultProcessor(BaseProcessor):
    """Default implementation of processor that wraps a worker processor.

    Validates that worker conforms to basic required invariants.
    """

    def __init__(self, worker_processor: Any):
        """Initializes the processor."""
        if worker_processor is None:
            raise ValueError("Worker processor must be provided!")
        elif not callable(worker_processor):
            raise ValueError("Worker processor is not callable!")
        elif not (
            hasattr(self._worker_processor, "apply_chat_template")
            and self._worker_processor.apply_chat_template is not None
            and callable(self._worker_processor.apply_chat_template)
        ):
            raise ValueError(
                "Worker processor doesn't have " "the `apply_chat_template` method"
            )

        self._worker_processor: Callable = worker_processor

        if not (
            hasattr(self._worker_processor, "tokenizer")
            and self._worker_processor.tokenizer is not None
        ):
            raise ValueError("Worker processor doesn't have a tokenizer!")

        if not isinstance(self._worker_processor.tokenizer, BaseTokenizer):
            raise ValueError(
                "Worker processor's tokenizer has unsupported type: "
                f"{type(self._worker_processor.tokenizer)}"
            )
        self._tokenizer: BaseTokenizer = self._worker_processor.tokenizer

        self._image_processor: Optional[BaseImageProcessor] = None
        if (
            hasattr(self._worker_processor, "image_processor")
            and self._worker_processor.image_processor is not None
        ):
            self._image_processor = DefaultImageProcessor(
                self._worker_processor.image_processor
            )

    @property
    @override
    def tokenizer(self) -> BaseTokenizer:
        """Returns a tokenizer associated with this processor."""
        return self._tokenizer

    @tokenizer.setter
    @override
    def tokenizer(self, new_tokenizer: BaseTokenizer) -> None:
        """Sets a tokenizer associated with this processor."""
        self._worker_processor.tokenizer = new_tokenizer
        self._tokenizer = new_tokenizer

    @property
    @override
    def chat_template(self) -> str:
        """Returns a chat template."""
        if not hasattr(self._worker_processor, "chat_template"):
            return ""
        return self._worker_processor.chat_template

    @chat_template.setter
    @override
    def chat_template(self, new_chat_template: str) -> None:
        """Sets a chat template associated with this processor."""
        self._worker_processor.chat_template = new_chat_template

    @property
    @override
    def image_processor(self) -> Optional[BaseImageProcessor]:
        """Returns an image processor."""
        return self._image_processor

    @property
    @override
    def image_token(self) -> Optional[str]:
        """Returns an image token."""
        if (
            hasattr(self._worker_processor, "image_token")
            and self._worker_processor.image_token
        ):
            return str(self._worker_processor.image_token)
        return None

    @property
    @override
    def image_token_id(self) -> Optional[int]:
        """Returns an image token id."""
        token_str = self.image_token
        if not token_str:
            return None
        image_token_id = self._tokenizer.convert_tokens_to_ids(token_str)  # type: ignore
        return int(image_token_id)

    @abc.abstractmethod
    def __call__(
        self,
        *,
        text: Union[str, List[str]],
        images: Union[PIL.Image.Image, List[PIL.Image.Image]],
        padding: bool,
        return_tensors: str = "pt",
    ) -> Dict[str, Any]:
        """Invokes the processor to extract features."""
        result = self._worker_processor(
            text=text, images=images, padding=padding, return_tensors=return_tensors
        )
        if result is None:
            raise RuntimeError("Processor returned `None`.")
        elif not isinstance(result, dict):
            raise RuntimeError(
                "Processor returned an object that is not a dictionary. "
                f"Actual type: {type(result)}"
            )
        return result

    @abc.abstractmethod
    def apply_chat_template(
        self, texts: Union[str, List[str]], add_generation_prompt: bool = False
    ) -> str:
        """Applies chat template."""
        result = self._worker_processor.apply_chat_template(
            texts=texts, add_generation_prompt=add_generation_prompt
        )
        if result is None:
            raise RuntimeError("`apply_chat_template` returned `None`.")
        elif not isinstance(result, str):
            raise RuntimeError(
                "`apply_chat_template` returned an object that is not a string. "
                f"Actual type: {type(result)}"
            )
        return result
