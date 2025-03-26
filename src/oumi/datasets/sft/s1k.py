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

@register_dataset("simplescaling/s1K")
class S1KDataset(BaseSftDataset):
    """Dataset class for the simplescaling/s1K dataset."""
    system_prompt = (
        "Below is the type of question, "
        "paired with the question that describes the problem. "
        "Write a response that gives appropriate answer to the question."
    )
    
    default_dataset = "simplescaling/s1K"
    QUERY_TEMPLATE_NOANSWER = """{Question}""".strip()

    def transform_conversation(self, example: Union[dict, pd.Series]) -> Conversation:
        """Transform a dataset example into a Conversation object."""

        thinking_trajectory: str = example.get("thinking_trajectories", None) or ""
        cot_type: str = example.get("cot_type", None) or ""
        question: str = example.get("question", None) or ""
        answer: str = example.get("attempt", None) or ""
        user_prompt = f"### Question Type:\n{cot_type}\n\n### Question:\n{question}"
        answer = "Answer: " + answer if "Answer:" not in answer else answer
        
        messages = [
            Message(role=Role.USER, content=user_prompt),
            Message(role=Role.ASSISTANT, content="<|im_start|>think\n" + "\n".join(thinking_trajectory).strip() + "\n<|im_start|>answer\n" + answer.strip())
        ]

        return Conversation(messages=messages)
