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

from pathlib import Path
from typing import Any, Callable, Optional, Union

import PIL.Image
import transformers
from typing_extensions import override

from oumi.core.processors.base_processor import BaseProcessor
from oumi.core.tokenizers.base_tokenizer import BaseTokenizer
from oumi.core.types.conversation import Message
from oumi.utils.str_utils import truncate_to_max_tokens_limit


class DefaultProcessor(BaseProcessor):
    """Thin wrapper around HuggingFace AutoProcessor providing Oumi conveniences.

    This class adds:
    - Oumi Message format support for apply_chat_template
    - label_ignore_index tracking for training
    - ignore_features list for filtering processor outputs
    """

    def __init__(
        self,
        processor_name: str,
        hf_processor: Any,
        tokenizer: BaseTokenizer,
        *,
        label_ignore_index: Optional[int],
        ignore_features: Optional[list[str]] = None,
    ):
        """Initializes the processor.

        Args:
            processor_name: Name of the processor/model.
            hf_processor: HuggingFace processor (from AutoProcessor.from_pretrained).
            tokenizer: Tokenizer associated with this processor.
            label_ignore_index: Special label value for loss computation exclusion.
            ignore_features: List of feature keys to remove from processor output.
        """
        if not processor_name:
            raise ValueError("Empty model name is not allowed!")

        self._processor_name = processor_name
        self._hf_processor: Callable = hf_processor
        self._tokenizer: BaseTokenizer = tokenizer
        self._label_ignore_index: Optional[int] = label_ignore_index
        self._ignore_features: list[str] = ignore_features or []

        # Sync tokenizer with HF processor
        if hasattr(self._hf_processor, "tokenizer"):
            self._hf_processor.tokenizer = tokenizer

        # Sync chat template
        if hasattr(self._hf_processor, "chat_template"):
            if (
                not self._hf_processor.chat_template
                or self._hf_processor.chat_template != tokenizer.chat_template
            ):
                self._hf_processor.chat_template = tokenizer.chat_template

    @property
    @override
    def processor_name(self) -> str:
        """Returns the processor name."""
        return self._processor_name

    @property
    @override
    def tokenizer(self) -> BaseTokenizer:
        """Returns the tokenizer associated with this processor."""
        return self._tokenizer

    @tokenizer.setter
    @override
    def tokenizer(self, new_tokenizer: BaseTokenizer) -> None:
        """Sets the tokenizer associated with this processor."""
        self._tokenizer = new_tokenizer
        if hasattr(self._hf_processor, "tokenizer"):
            self._hf_processor.tokenizer = new_tokenizer

    @property
    @override
    def chat_template(self) -> str:
        """Returns the chat template."""
        if hasattr(self._hf_processor, "chat_template"):
            return self._hf_processor.chat_template or ""
        return ""

    @chat_template.setter
    @override
    def chat_template(self, new_chat_template: str) -> None:
        """Sets the chat template."""
        if hasattr(self._hf_processor, "chat_template"):
            self._hf_processor.chat_template = new_chat_template

    @property
    @override
    def image_processor(self) -> Optional[transformers.ImageProcessingMixin]:
        """Returns the image processor."""
        return getattr(self._hf_processor, "image_processor", None)

    @property
    @override
    def image_token(self) -> Optional[str]:
        """Returns the image token string."""
        if hasattr(self._hf_processor, "image_token"):
            token = self._hf_processor.image_token
            if token:
                return str(token)
        return None

    @property
    @override
    def image_token_id(self) -> Optional[int]:
        """Returns the image token ID."""
        token_str = self.image_token
        if not token_str:
            return None

        token_id = self._tokenizer.convert_tokens_to_ids(token_str)  # type: ignore
        if not isinstance(token_id, int):
            raise ValueError(
                f"Image token '{token_str}' not in tokenizer vocabulary. "
                f"Got type {type(token_id)} instead of int."
            )
        return token_id

    @property
    @override
    def label_ignore_index(self) -> Optional[int]:
        """Returns the label ignore index for loss computation."""
        return self._label_ignore_index

    @property
    @override
    def ignore_features(self) -> list[str]:
        """Returns the list of features to ignore."""
        return self._ignore_features.copy()

    @property
    @override
    def raw_processor(self) -> Callable:
        """Returns the underlying HuggingFace processor."""
        return self._hf_processor

    @override
    def __call__(
        self,
        *,
        text: list[str],
        images: Optional[list[PIL.Image.Image]] = None,
        return_tensors: Optional[str] = "pt",
        **kwargs,
    ) -> transformers.BatchEncoding:
        """Invokes the processor to extract features.

        Args:
            text: List of text prompts.
            images: Optional list of input images.
            return_tensors: Format of returned tensors ('pt', 'np', etc).
            **kwargs: Additional arguments passed to HF processor.

        Returns:
            BatchEncoding with model-specific input features.
        """
        # Direct delegation to HF processor
        if images is None or len(images) == 0:
            result = self._hf_processor(
                text=text,
                return_tensors=return_tensors,
                **kwargs,
            )
        else:
            # Single text string for single image, list for multiple
            text_input = text[0] if len(text) == 1 else text
            result = self._hf_processor(
                text=text_input,
                images=images,
                return_tensors=return_tensors,
                **kwargs,
            )

        # Remove ignored features if specified
        if self._ignore_features and isinstance(result, dict):
            for key in self._ignore_features:
                result.pop(key, None)

        # Ensure result is BatchEncoding
        if isinstance(result, transformers.BatchFeature):
            result = transformers.BatchEncoding(
                data=dict(**result), tensor_type=return_tensors
            )
        elif isinstance(result, dict):
            result = transformers.BatchEncoding(
                data=result, tensor_type=return_tensors
            )
        elif not isinstance(result, transformers.BatchEncoding):
            raise RuntimeError(
                f"Processor returned unexpected type: {type(result)}. "
                "Expected BatchEncoding."
            )

        return result

    @override
    def apply_chat_template(
        self, conversation: list[Message], add_generation_prompt: bool = False
    ) -> str:
        """Applies chat template to Oumi conversation format.

        This is the main Oumi-specific convenience - handling Oumi Message objects
        instead of requiring HF's dict format.

        Args:
            conversation: List of Oumi Message objects.
            add_generation_prompt: Whether to append generation prompt.

        Returns:
            Formatted text prompt string.
        """
        # Special handling for tokenizer-only processors
        if isinstance(self._hf_processor, BaseTokenizer):
            # Tokenizer can't handle images
            for message in conversation:
                if message.contains_images():
                    raise ValueError(
                        f"Message {message.id} contains images, but processor "
                        "is text-only (tokenizer). Use a multimodal processor."
                    )

            result = self._hf_processor.apply_chat_template(
                conversation,  # type: ignore
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        else:
            # Multimodal processor - wrap in list
            result = self._hf_processor.apply_chat_template(
                [conversation], add_generation_prompt=add_generation_prompt
            )

        # Unwrap single-item list
        if isinstance(result, list) and len(result) == 1:
            result = result[0]

        if not isinstance(result, str):
            raise RuntimeError(
                f"apply_chat_template returned {type(result)}, expected str."
            )

        return result

    @override
    def save_config(self, output_dir: Union[Path, str]) -> None:
        """Saves processor config to directory."""
        if hasattr(self._hf_processor, "save_pretrained"):
            self._hf_processor.save_pretrained(str(output_dir))

    @override
    def truncate_text(
        self,
        text: str,
        *,
        max_tokens: int,
        truncation_side: str = "right",
    ) -> tuple[str, int]:
        """Truncates text to max_tokens.

        Args:
            text: Text to truncate.
            max_tokens: Maximum number of tokens.
            truncation_side: Which side to truncate ("right" or "left").

        Returns:
            Tuple of (truncated_text, num_tokens).
        """
        return truncate_to_max_tokens_limit(
            text,
            self._tokenizer,
            max_tokens=max_tokens,
            truncation_side=truncation_side,
        )
