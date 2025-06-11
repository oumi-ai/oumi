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

"""Generic class for using HuggingFace vision-language datasets.

Allows users to specify the image, question, and answer columns at the config level.
"""

import base64
from typing import Any, Optional, Union

import pandas as pd
from typing_extensions import override

from oumi.core.datasets import VisionLanguageSftDataset
from oumi.core.registry import register_dataset
from oumi.core.types.conversation import (
    ContentItem,
    Conversation,
    Message,
    Role,
    Type,
)


@register_dataset("hf_vision")
class HuggingFaceVisionDataset(VisionLanguageSftDataset):
    """Converts HuggingFace Vision-Language Datasets to Oumi Message format.

    This dataset handles standard HuggingFace datasets that contain:
    - An image column (containing image data or paths)
    - A question/prompt column (text input)
    - An optional answer column (text output)

    Example:
        dataset = HuggingFaceVisionDataset(
            hf_dataset_path="HuggingFaceM4/VQAv2",
            image_column="image",
            question_column="question",
            answer_column="answer"
        )
    """

    def __init__(
        self,
        *,
        hf_dataset_path: str,
        image_column: str,
        question_column: str,
        answer_column: Optional[str] = None,
        system_prompt_column: Optional[str] = None,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Initializes a new instance of the HuggingFaceVisionDataset class.

        Args:
            hf_dataset_path: Path to the HuggingFace dataset.
            image_column: Name of the column containing image data.
            question_column: Name of the column containing the question/prompt text.
            answer_column: Optional name of the column containing the answer text.
            system_prompt: Optional system prompt to add as the first message.
            system_prompt_column: Optional name of the column containing system prompts.
            **kwargs: Additional arguments passed to the parent class.
        """
        if not hf_dataset_path:
            raise ValueError("The `hf_dataset_path` parameter must be provided.")
        if not image_column:
            raise ValueError("The `image_column` parameter must be provided.")
        if not question_column:
            raise ValueError("The `question_column` parameter must be provided.")

        self.image_column = image_column
        self.question_column = question_column
        self.answer_column = answer_column
        self.system_prompt = system_prompt
        self.system_prompt_column = system_prompt_column

        if system_prompt and system_prompt_column:
            raise ValueError(
                "Only one of `system_prompt` or `system_prompt_column` can be provided."
            )

        kwargs["dataset_name"] = hf_dataset_path
        super().__init__(**kwargs)

    def _get_image_content_item(self, image_data) -> ContentItem:
        """Create a ContentItem for the image data.

        Args:
            image_data: Image data from the dataset (could be bytes, PIL Image, etc.).

        Returns:
            ContentItem containing the image data.
        """
        if isinstance(image_data, bytes):
            # Raw bytes
            return ContentItem(
                type=Type.IMAGE_BINARY,
                binary=image_data,
            )
        elif hasattr(image_data, "bytes"):
            # PIL Image or similar object with bytes attribute
            return ContentItem(
                type=Type.IMAGE_BINARY,
                binary=image_data.bytes,
            )
        elif isinstance(image_data, dict) and "bytes" in image_data:
            # Dict with bytes
            return ContentItem(
                type=Type.IMAGE_BINARY,
                binary=image_data["bytes"],
            )
        elif isinstance(image_data, str):
            if image_data.startswith(("http://", "https://")):
                return ContentItem(type=Type.IMAGE_URL, content=image_data)
            else:
                # Assume it's a base64 image
                return ContentItem(
                    type=Type.IMAGE_BINARY, binary=base64.b64decode(image_data)
                )
        else:
            raise ValueError(
                f"Unsupported image data type: {type(image_data)}. "
                "Expected bytes, PIL Image, URL string, or base64 encoded string."
            )

    @override
    def transform_conversation(self, example: Union[dict, pd.Series]) -> Conversation:
        """Preprocesses the inputs of the example and returns a Conversation.

        Args:
            example: An example containing image, question, and optionally answer data.

        Returns:
            Conversation: A Conversation object containing the messages.
        """
        messages = []

        # Validate required columns
        required = {
            "image_column": self.image_column,
            "question_column": self.question_column,
        }

        if self.answer_column:
            required["answer_column"] = self.answer_column

        if self.system_prompt_column:
            required["system_prompt_column"] = self.system_prompt_column

        for column_name, column_var in required.items():
            if column_var not in example:
                raise ValueError(
                    f"The column '{column_name}' (specified as {column_var}) "
                    f"is not present in the example. "
                    f"Available columns: {list(example.keys())}"
                )

        # Add system prompt if available (either static or from column)
        if self.system_prompt:
            system_message_content = self.system_prompt
        else:
            system_message_content = self._process_text_value(
                example.get(self.system_prompt_column)
            )

        if system_message_content:
            messages.append(Message(role=Role.SYSTEM, content=system_message_content))

        # Extract and process the data
        image_data = example[self.image_column]
        question_text = self._process_text_value(example[self.question_column])

        # Create the image content item
        image_content_item = self._get_image_content_item(image_data)

        # Create the user message with image and text
        user_message = Message(
            role=Role.USER,
            content=[
                image_content_item,
                ContentItem(type=Type.TEXT, content=question_text),
            ],
        )
        messages.append(user_message)

        # Add assistant response if answer column is specified and present
        if self.answer_column and self.answer_column in example:
            answer_text = self._process_text_value(example[self.answer_column])
            assistant_message = Message(role=Role.ASSISTANT, content=answer_text)
            messages.append(assistant_message)

        return Conversation(messages=messages)

    def _process_text_value(self, s: Any) -> str:
        """Process a text value.

        Args:
            s: The text value to process.

        Returns:
            The processed text value.
        """
        if s is None:
            return ""
        if isinstance(s, str):
            # The data contains occasional `\n` at the beginning or end
            # of text values. Let's strip them.
            return s.strip()
        raise ValueError(f"Unsupported text value type: {type(s)}")
