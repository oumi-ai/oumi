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


@register_dataset("CohereForAI/aya_dataset")
class AyaDataset(BaseSftDataset):
    """Dataset class for the CohereForAI/aya_dataset dataset."""

    default_dataset = "CohereForAI/aya_dataset"

    def transform_conversation(self, example: Union[dict, pd.Series]) -> Conversation:
        """Transform a dataset example into a Conversation object."""
        question: str = example.get("inputs", None) or ""
        answer: str = example.get("targets", None) or ""

        messages = [
            Message(role=Role.USER, content=question),
            Message(role=Role.ASSISTANT, content=answer),
        ]

        return Conversation(messages=messages)
