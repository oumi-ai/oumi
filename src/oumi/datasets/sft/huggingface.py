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

"""Generic class for using HuggingFace datasets with messages column.

Allows users to specify the messages column at the config level.
"""

from typing import Union

import pandas as pd

from oumi.core.datasets import BaseSftDataset
from oumi.core.registry import register_dataset
from oumi.core.types.conversation import Conversation, Message, Role


@register_dataset("HuggingFaceDataset")
class HuggingFaceDataset(BaseSftDataset):
    """Converts HuggingFace Datasets with messages to Oumi Message format.

    Example:
        dataset = HuggingFaceDataset(
            hf_dataset_path="oumi-ai/oumi-synthetic-document-claims",
            message_column="messages"
        )
    """

    def __init__(
        self,
        *,
        hf_dataset_path: str = "",
        messages_column: str = "messages",
        exclude_final_assistant_message: bool = False,
        **kwargs,
    ) -> None:
        """Initializes a new instance of the OumiDataset class."""
        if not hf_dataset_path:
            raise ValueError("The `hf_dataset_path` parameter must be provided.")
        if not messages_column:
            raise ValueError("The `messages_column` parameter must be provided.")
        self.messages_column = messages_column
        self.exclude_final_assistant_message = exclude_final_assistant_message
        kwargs["dataset_name"] = hf_dataset_path
        super().__init__(**kwargs)

    def transform_conversation(self, example: Union[dict, pd.Series]) -> Conversation:
        """Preprocesses the inputs of the example and returns a dictionary.

        Args:
            example: An example containing `messages` entries.

        Returns:
            Conversation: A Conversation object containing the messages.
        """
        messages = []

        if self.messages_column not in example:
            raise ValueError(
                f"The column '{self.messages_column}' is not present in the example."
            )
        example_messages = example[self.messages_column]

        for message in example_messages:
            if "role" not in message or "content" not in message:
                raise ValueError(
                    "The message format is invalid. Expected keys: 'role', 'content'."
                )
            if message["role"] == "user":
                role = Role.USER
            elif message["role"] == "assistant":
                role = Role.ASSISTANT
            else:
                raise ValueError(
                    f"Invalid role '{message['role']}'. Expected 'user' or 'assistant'."
                )
            content = message["content"] or ""
            messages.append(Message(role=role, content=content))

        if self.exclude_final_assistant_message and messages[-1].role == Role.ASSISTANT:
            messages = messages[:-1]

        return Conversation(messages=messages)
