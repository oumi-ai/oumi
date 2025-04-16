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

"""Generic class for using HuggingFace datasets with input/output columns.

Allows users to specify the prompt and response columns at the config level.
"""

from typing import Union

import pandas as pd

from oumi.core.datasets import BaseSftDataset
from oumi.core.registry import register_dataset
from oumi.core.types.conversation import Conversation, Message, Role


@register_dataset("PromptResponseDataset")
class PromptResponseDataset(BaseSftDataset):
    """Converts HuggingFace Datasets with input/output columns to Message format.

    Example:
        dataset = PromptResponseDataset(hf_dataset_path="O1-OPEN/OpenO1-SFT",
        prompt_column="instruction",
        response_column="output")
    """

    default_dataset = "O1-OPEN/OpenO1-SFT"

    def __init__(
        self,
        *,
        hf_dataset_path: str = "O1-OPEN/OpenO1-SFT",
        prompt_column: str = "instruction",
        response_column: str = "output",
        **kwargs,
    ) -> None:
        """Initializes a new instance of the PromptResponseDataset class."""
        self.prompt_column = prompt_column
        self.response_column = response_column
        kwargs["dataset_name"] = hf_dataset_path
        super().__init__(**kwargs)

    def transform_conversation(self, example: Union[dict, pd.Series]) -> Conversation:
        """Preprocesses the inputs of the example and returns a dictionary.

        Args:
            example (dict or Pandas Series): An example containing `input` (optional),
                `instruction`, and `output` entries.

        Returns:
            dict: The input example converted to messages dictionary format.

        """
        messages = []

        user_prompt = str(example[self.prompt_column])
        messages.append(Message(role=Role.USER, content=user_prompt))

        if self.response_column:
            model_output = str(example[self.response_column])
            messages.append(Message(role=Role.ASSISTANT, content=model_output))

        return Conversation(messages=messages)
