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
from oumi.core.types.conversation import Conversation, Message


@register_dataset("HuggingFaceH4/ultrachat_200k")
class UltrachatH4Dataset(BaseSftDataset):
    """Dataset class for the HuggingFaceH4/ultrachat_200k dataset."""

    default_dataset = "HuggingFaceH4/ultrachat_200k"

    def transform_conversation(self, example: Union[dict, pd.Series]) -> Conversation:
        """Transform a dataset example into a Conversation object."""
        raw_messages = example.get("messages")
        if raw_messages is None:
            raise ValueError("Invalid messages")

        messages = [Message.model_validate(message) for message in raw_messages]
        return Conversation(messages=messages)
