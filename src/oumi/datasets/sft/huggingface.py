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
    """Converts a HuggingFace dataset to Oumi message format.

    This class supports HF datasets in two formats:

    1) Messages format:
        Each example contains a `messages_column` in the following format:
        [
            {'role': 'user', 'content': ...},
            {'role': 'assistant', 'content': ...}
        ]

    Sample code to load the dataset:
        dataset = HuggingFaceDataset(
            hf_dataset_path="oumi-ai/oumi-synthetic-document-claims",
            message_column="messages",
            split="validation",
        )

    2) Prompt format:
        Each example contains a `prompt_column` that corresponds to a user prompt:
        <the prompt to an assistant>

    Sample code:
        dataset = HuggingFaceDataset(
            hf_dataset_path="oumi-ai/oumi-document-hallucination-benchmark",
            prompt_column="gpt_4o prompt",
            split="test",
        )
    """

    def __init__(
        self,
        *,
        hf_dataset_path: str = "",
        prompt_column: str = "",
        messages_column: str = "messages",
        exclude_final_assistant_message: bool = False,
        **kwargs,
    ) -> None:
        """Initializes a new instance of the OumiDataset class."""
        if not hf_dataset_path:
            raise ValueError("The `hf_dataset_path` parameter must be provided.")
        if not messages_column and not prompt_column:
            raise ValueError(
                "Either the `messages_column` parameter or the `prompt_column`"
                "parameter must be provided."
            )
        self.prompt_column = prompt_column
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
        # Prompt format.
        if self.prompt_column:
            if self.prompt_column not in example:
                raise ValueError(
                    f"The column '{self.prompt_column}' is not present in the example."
                )
            prompt = str(example[self.prompt_column])
            return Conversation(messages=[Message(role=Role.USER, content=prompt)])

        # Messages format.
        if self.messages_column not in example:
            raise ValueError(
                f"The column '{self.messages_column}' is not present in the example."
            )
        example_messages = example[self.messages_column]

        oumi_messages = []
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
            oumi_messages.append(Message(role=role, content=content))

        if (
            self.exclude_final_assistant_message
            and oumi_messages[-1].role == Role.ASSISTANT
        ):
            oumi_messages = oumi_messages[:-1]

        return Conversation(messages=oumi_messages)
