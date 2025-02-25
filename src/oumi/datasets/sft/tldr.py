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

from oumi.core.datasets.base_sft_dataset import BaseSftDataset
from oumi.core.registry import register_dataset
from oumi.core.types.conversation import Conversation, Message, Role


@register_dataset("trl-lib/tldr_sft")
class TldrSftDataset(BaseSftDataset):
    default_dataset = "trl-lib/tldr"

    _DEFAULT_SYSTEM_PROMPT = (
        "You are an expert summarizer. "
        "Your task is to distill any given text into a concise, "
        'easily understandable "TL;DR" (Too Long; Didn\'t Read) summary.'
    )

    def __init__(
        self,
        **kwargs,
    ) -> None:
        """Initializes a new instance of the BaseSftDataset class."""
        tokenize = kwargs.pop("tokenize", False)
        text_col = kwargs.pop("text_col", "prompt")
        super().__init__(
            tokenize=tokenize,
            text_col=text_col,
            **kwargs,
        )

    def transform_conversation(self, example: Union[dict, pd.Series]) -> Conversation:
        """Transform a dataset example into a Conversation object."""
        prompt: str = example.get("prompt", None) or ""
        completion: str = example.get("completion", None) or ""

        messages = [
            Message(role=Role.SYSTEM, content=self._DEFAULT_SYSTEM_PROMPT),
            Message(role=Role.USER, content=prompt),
            Message(role=Role.ASSISTANT, content=completion),
        ]

        return Conversation(messages=messages)
