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

import copy
from pathlib import Path
from typing import Any, Callable, Optional, Union

import PIL.Image
import transformers
from typing_extensions import override

from oumi.core.processors.base_image_processor import BaseImageProcessor
from oumi.core.processors.base_processor import BaseProcessor
from oumi.core.processors.default_image_processor import DefaultImageProcessor
from oumi.core.tokenizers.base_tokenizer import BaseTokenizer
from oumi.core.types.conversation import Message
from oumi.utils.logging import logger
from oumi.utils.str_utils import truncate_to_max_tokens_limit


class DefaultProcessor(BaseProcessor):
    """Default implementation of processor that wraps a worker processor.

    Validates that worker conforms to basic required invariants.
    """

    def __init__(
        self,
        processor_name: str,
        worker_processor: Any,
        tokenizer: BaseTokenizer,
        *,
        label_ignore_index: Optional[int],
        ignore_features: Optional[list[str]] = None,
    ):
        """Initializes the processor."""
        if not processor_name:
            raise ValueError("Processor name must be provided!")
        elif worker_processor is None:
            raise ValueError("Worker processor must be provided!")
        elif not callable(worker_processor):
            raise ValueError("Worker processor is not callable!")
        elif not (
            hasattr(worker_processor, "apply_chat_template")
            and worker_processor.apply_chat_template is not None
            and callable(worker_processor.apply_chat_template)
        ):
            raise ValueError(
                "Worker processor doesn't have the `apply_chat_template` method"
            )

        self._processor_name = processor_name
        self._worker_processor: Callable = worker_processor
        self._worker_processor.tokenizer = tokenizer
        self._tokenizer: BaseTokenizer = tokenizer

        # If the worker processor does not have a chat template, or has a different
        # one, then equate it to tokenizer's.
        if (
            not hasattr(self._worker_processor, "chat_template")
            or self._worker_processor.chat_template != tokenizer.chat_template
        ):
            self._worker_processor.chat_template = tokenizer.chat_template

        self._image_processor: Optional[BaseImageProcessor] = None
        if (
            hasattr(self._worker_processor, "image_processor")
            and self._worker_processor.image_processor is not None
        ):
            self._image_processor = DefaultImageProcessor(
                self._worker_processor.image_processor
            )
        self._label_ignore_index: Optional[int] = label_ignore_index
        self._ignore_features: Optional[list[str]] = (
            copy.copy(ignore_features) if ignore_features else []
        )

    @property
    @override
    def processor_name(self) -> str:
        """Returns a processor name."""
        return self._processor_name

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

        token_id = self._tokenizer.convert_tokens_to_ids(token_str)  # type: ignore
        if not isinstance(token_id, int):
            raise ValueError(
                "Image token id must be an integer. "
                "The token is likely not in tokenizer's vocabulary. "
                f"Image token: '{token_str}' "
                f"Actual type: {type(token_id)}"
            )
        return int(token_id)

    @property
    @override
    def label_ignore_index(self) -> Optional[int]:
        """Returns a label ignore index."""
        return self._label_ignore_index

    @property
    @override
    def ignore_features(self) -> list[str]:
        """Returns a list of keys of features to ignore from feeding the model."""
        return copy.copy(self._ignore_features) if self._ignore_features else []

    @property
    @override
    def raw_processor(self) -> Callable:
        """Returns the underlying raw processor."""
        return self._worker_processor

    @override
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
        if images is None or len(images) == 0:
            result = self._worker_processor(
                text=text, padding=padding, return_tensors=return_tensors
            )
        else:
            result = self._worker_processor(
                text=(text[0] if len(text) == 1 else text),
                images=images,
                padding=padding,
                return_tensors=return_tensors,
            )
        if result is None:
            raise RuntimeError("Processor returned `None`.")
        elif isinstance(result, transformers.BatchFeature):
            for key in self.ignore_features:
                if key in result:  # If it is not, we do not act to allow
                    del result[key]  # the underlying dataset/processor to vary
                    # their examples/output across batches.

            result = transformers.BatchEncoding(
                data=dict(**result), tensor_type=return_tensors
            )
        elif isinstance(result, dict):
            result = transformers.BatchEncoding(data=result, tensor_type=return_tensors)
        elif not isinstance(result, transformers.BatchEncoding):
            raise RuntimeError(
                "Processor returned an object that is not a BatchEncoding. "
                f"Actual type: {type(result)}"
            )
        return result

    @override
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
        if isinstance(self._worker_processor, BaseTokenizer):
            # If the processor is actually a tokenizer, then disallow non-text messages.
            for message in conversation:
                if message.contains_images():
                    raise ValueError(
                        f"Conversation includes non-text messages: {message.id}. "
                        "This is not allowed for processors that are tokenizers."
                    )

            result = self._worker_processor.apply_chat_template(
                conversation,  # type: ignore
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        else:
            result = self._worker_processor.apply_chat_template(
                conversation, add_generation_prompt=add_generation_prompt
            )

        if result is None:
            raise RuntimeError("`apply_chat_template` returned `None`.")
        elif not isinstance(result, str):
            raise RuntimeError(
                "`apply_chat_template` returned an object that is not a string. "
                f"Actual type: {type(result)}"
            )
        return result

    @override
    def save_config(self, output_dir: Union[Path, str]) -> None:
        """Saves processor config to the directory."""
        if not (
            hasattr(self._worker_processor, "save_pretrained")
            and self._worker_processor.save_pretrained is not None
            and callable(self._worker_processor.save_pretrained)
        ):
            logger.warning(
                "Don't know how to save processor config "
                f"to output dir: {output_dir}. "
                "Ignored the request!"
            )
            return

        self._worker_processor.save_pretrained(str(output_dir))

    @override
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
        return truncate_to_max_tokens_limit(
            text,
            self._tokenizer,
            max_tokens=max_tokens,
            truncation_side=truncation_side,
        )
