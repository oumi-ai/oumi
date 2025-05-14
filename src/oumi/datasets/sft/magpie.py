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

from typing import Union

import pandas as pd

from oumi.core.datasets import BaseSftDataset
from oumi.core.registry import register_dataset
from oumi.core.types.conversation import Conversation, Message, Role


@register_dataset("argilla/magpie-ultra-v0.1")
class ArgillaMagpieUltraDataset(BaseSftDataset):
    """Dataset class for the argilla/magpie-ultra-v0.1 dataset."""

    default_dataset = "argilla/magpie-ultra-v0.1"

    def transform_conversation(self, example: Union[dict, pd.Series]) -> Conversation:
        """Transform a dataset example into a Conversation object."""
        instruction: str = example.get("instruction", None) or ""
        response: str = example.get("response", None) or ""

        messages = [
            Message(role=Role.USER, content=instruction),
            Message(role=Role.ASSISTANT, content=response),
        ]

        return Conversation(messages=messages)


@register_dataset("Magpie-Align/Llama-3-Magpie-Pro-1M-v0.1")
@register_dataset("Magpie-Align/Magpie-Pro-300K-Filtered")
class MagpieProDataset(BaseSftDataset):
    """Dataset class for the Magpie-Align/Llama-3-Magpie-Pro-1M-v0.1 dataset."""

    default_dataset = "Magpie-Align/Llama-3-Magpie-Pro-1M-v0.1"

    def transform_conversation(self, example: Union[dict, pd.Series]) -> Conversation:
        """Transform a dataset example into a Conversation object."""
        conversation = example.get("conversations")

        if conversation is None:
            raise ValueError("Conversation is None")

        messages = []
        for message in conversation:
            if message["from"] == "human":
                role = Role.USER
            elif message["from"] == "gpt":
                role = Role.ASSISTANT
            else:
                raise ValueError(f"Unknown role: {message['from']}")
            content = message.get("value", "")
            messages.append(Message(role=role, content=content))
        return Conversation(messages=messages)
