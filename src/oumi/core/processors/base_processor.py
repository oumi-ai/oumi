# Copyright 2025 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
from pathlib import Path
from typing import Callable, Optional, Union

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
    def processor_name(self) -> str:
        """Returns a processor name."""
        raise NotImplementedError

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

    @property
    @abc.abstractmethod
    def label_ignore_index(self) -> Optional[int]:
        """Returns a label ignore index."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def ignore_features(self) -> list[str]:
        """Returns a list of keys of features to ignore from feeding the model."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def raw_processor(self) -> Callable:
        """Returns the underlying raw processor.

        The use of this method is generally discouraged. Only use it if you know
        what you are doing e.g., direct access to the underlying processor
        is required by some third-party library.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(
        self,
        *,
        text: list[str],
        padding: bool,
        images: Optional[list[PIL.Image.Image]] = None,
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
        self, conversation: list[Message], add_generation_prompt: bool = False
    ) -> str:
        """Applies a chat template.

        Args:
            conversation: A list of messages (conversation "turns").
            add_generation_prompt: Whether to append generation prompt to the output.

        Returns:
            A text prompt, which includes all input messages formatted into a string.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def save_config(self, output_dir: Union[Path, str]) -> None:
        """Saves processor config to the directory."""
        raise NotImplementedError

    @abc.abstractmethod
    def truncate_text(
        self,
        text: str,
        *,
        max_tokens: int,
        truncation_side: str = "right",
    ) -> tuple[str, int]:
        """Truncates text to `max_length` in tokens.

        Args:
            text: A text prompt.
            max_tokens: Maximum number of tokens to keep.
            truncation_side: The side to truncate the tokens ("right" or "left").

        Returns:
            A tuple containing truncated text prompt and the number of tokens.
        """
        raise NotImplementedError
