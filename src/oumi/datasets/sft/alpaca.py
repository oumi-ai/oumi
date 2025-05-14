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

"""Porting the Alpaca dataset with Oumi.

For more info see:
    (1) https://github.com/tatsu-lab/stanford_alpaca
    (2) https://github.com/gururise/AlpacaDataCleaned
"""

from typing import Union, cast

import pandas as pd

from oumi.core.datasets import BaseSftDataset
from oumi.core.registry import register_dataset
from oumi.core.types.conversation import Conversation, Message, Role


@register_dataset("yahma/alpaca-cleaned")
@register_dataset("tatsu-lab/alpaca")
class AlpacaDataset(BaseSftDataset):
    system_prompt_with_context = (
        "Below is an instruction that describes a task, "
        "paired with an input that provides further context. "
        "Write a response that appropriately completes the request."
    )

    system_prompt_without_context = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
    )

    default_dataset = "tatsu-lab/alpaca"

    def __init__(
        self,
        *,
        include_system_prompt: bool = True,
        **kwargs,
    ) -> None:
        """Initializes a new instance of the AlpacaDataset class."""
        self.include_system_prompt = include_system_prompt

        super().__init__(**kwargs)

    def transform_conversation(self, example: Union[dict, pd.Series]) -> Conversation:
        """Preprocesses the inputs of the example and returns a dictionary.

        Args:
            example (dict or Pandas Series): An example containing `input` (optional),
                `instruction`, and `output` entries.

        Returns:
            dict: The input example converted to Alpaca dictionary format.

        """
        messages = []

        # Use default Alpaca user prompt template
        if ("input" in example) and len(example["input"]) > 0:
            # This example has both an instruction and a user input.
            user_prompt = f"{example['instruction']}\n\n### Input:\n{example['input']}"
            system_prompt = self.system_prompt_with_context
        else:
            user_prompt = cast(str, example["instruction"])
            system_prompt = self.system_prompt_without_context

        model_output = cast(str, example["output"])

        # Create message list
        if self.include_system_prompt:
            messages.append(Message(role=Role.SYSTEM, content=system_prompt))
        messages.append(Message(role=Role.USER, content=user_prompt))
        messages.append(Message(role=Role.ASSISTANT, content=model_output))

        return Conversation(messages=messages)
