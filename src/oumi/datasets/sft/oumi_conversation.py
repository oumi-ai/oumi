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


@register_dataset("OumiConversationDataset")
class OumiConversationDataset(BaseSftDataset):
    """Converts HuggingFace Datasets with messages to Oumi Message format.

    Example:
        dataset = OumiConversationDataset(
            hf_dataset_path="oumi-ai/oumi-synthetic-document-claims",
            message_column="messages"
        )
    """

    default_dataset = "oumi-ai/oumi-synthetic-document-claims"

    def __init__(
        self,
        *,
        hf_dataset_path: str = "oumi-ai/oumi-synthetic-document-claims",
        messages_column: str = "messages",
        exclude_final_assistant_message: bool = False,
        **kwargs,
    ) -> None:
        """Initializes a new instance of the OumiDataset class."""
        self.messages_column = messages_column
        self.exclude_final_assistant_message = exclude_final_assistant_message
        kwargs["dataset_name"] = hf_dataset_path
        super().__init__(**kwargs)

    def transform_conversation(self, example: Union[dict, pd.Series]) -> Conversation:
        """Preprocesses the inputs of the example and returns a dictionary.

        Args:
            example (dict or Pandas Series): An example containing `messages` entries.

        Returns:
            dict: The input example converted to messages dictionary format.

        """
        messages = []

        example_messages = example[self.messages_column]
        for message in example_messages:
            role = Role.USER if message["role"] == "user" else Role.ASSISTANT
            content = message["content"]
            messages.append(Message(role=role, content=content))

        if self.exclude_final_assistant_message:
            if messages[-1].role == Role.ASSISTANT:
                messages = messages[:-1]

        return Conversation(messages=messages)
