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

"""Porting the Alpaca evaluation dataset to Oumi.

For more info see: https://github.com/tatsu-lab/alpaca_eval
"""

from typing import Union, cast

import pandas as pd

from oumi.core.datasets import BaseSftDataset
from oumi.core.registry import register_dataset
from oumi.core.types.conversation import Conversation, Message, Role


@register_dataset("tatsu-lab/alpaca_eval")
class AlpacaEvalDataset(BaseSftDataset):
    system_prompt_with_context = (
        "Below is an instruction that describes a task, "
        "paired with an input that provides further context. "
        "Write a response that appropriately completes the request."
    )

    system_prompt_without_context = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
    )

    default_dataset = "tatsu-lab/alpaca_eval"

    def __init__(
        self,
        *,
        include_system_prompt: bool = False,
        unused_entries_to_metadata: bool = False,
        trust_remote_code: bool = True,
        **kwargs,
    ) -> None:
        """Initializes a new instance of the AlpacaEvalDataset class.

        Args:
            include_system_prompt: Whether to include a system prompt in the
                conversation.
            unused_entries_to_metadata (bool): Whether to save entries that were not
                used in the conversation (entries other than `instruction`, `input`)
                as metadata.
            trust_remote_code: Whether to trust remote code.
            **kwargs: Additional keyword arguments.
        """
        self.include_system_prompt = include_system_prompt
        self.unused_entries_to_metadata = unused_entries_to_metadata

        super().__init__(**kwargs, trust_remote_code=trust_remote_code)

    def transform_conversation(self, example: Union[dict, pd.Series]) -> Conversation:
        """Preprocesses the inputs of the example and returns a dictionary.

        Args:
            example (dict or Pandas Series): An example containing `input` (optional),
                `instruction` entries.

        Returns:
            dict: The input example converted to Alpaca dictionary format.

        Note:
            If `unused_entries_to_metadata` is set: all example's entries, other than
            the expected ones (i.e., `input` and `instruction`), are saved as metadata.
        """
        messages = []

        # Use default Alpaca user prompt template.
        if ("input" in example) and len(example["input"]) > 0:
            # This example has both an instruction and a user input.
            user_prompt = f"{example['instruction']}\n\n### Input:\n{example['input']}"
            system_prompt = self.system_prompt_with_context
        else:
            user_prompt = cast(str, example["instruction"])
            system_prompt = self.system_prompt_without_context

        # Create message list.
        if self.include_system_prompt:
            messages.append(Message(role=Role.SYSTEM, content=system_prompt))
        messages.append(Message(role=Role.USER, content=user_prompt))

        # Retain entries (other than `instruction`, `input`) as metadata.
        metadata_fields = set()
        if self.unused_entries_to_metadata:
            if isinstance(example, pd.Series):
                metadata_fields = {str(i) for i in example.index}
            elif isinstance(example, dict):
                metadata_fields = {str(key) for key in example.keys()}
            metadata_fields = metadata_fields - {"instruction", "input"}
        metadata = {field: example[field] for field in metadata_fields}

        return Conversation(messages=messages, metadata=metadata)
