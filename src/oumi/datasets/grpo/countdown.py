# Copyright 2025 - Jiayi Pan
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

"""Derived from https://github.com/Jiayi-Pan/TinyZero/blob/main/examples/data_preprocess/countdown.py.

This file was slightly modified to inherit from the `BaseExperimentalGrpoDataset` class.
"""

import pandas as pd
from typing_extensions import override

from oumi.core.datasets.base_grpo_dataset import BaseExperimentalGrpoDataset
from oumi.core.registry import register_dataset
from oumi.core.types.conversation import Conversation

_PROMPT_TEMPLATE = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
User: Using the numbers {nums}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.
Assistant: Let me solve this step by step.
<think>"""  # noqa: E501


@register_dataset("d1shs0ap/countdown")
class CountdownGrpoDataset(BaseExperimentalGrpoDataset):
    """Dataset class for the `d1shs0ap/countdown` dataset.

    A sample from the dataset:
    {"target": 87, "nums": [79, 8]}
    """

    default_dataset = "d1shs0ap/countdown"

    @override
    def transform(self, sample: pd.Series) -> dict:
        """Validate and transform the sample into Python `dict`."""
        # Convert nums from np array to list
        target = int(sample["target"])
        nums = [int(num) for num in sample["nums"]]
        prompt = _PROMPT_TEMPLATE.format(nums=nums, target=target)
        solution = {
            "target": target,
            "numbers": nums,
        }
        return {
            "data_source": "countdown",
            "prompt": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": solution},
            "extra_info": {
                "split": self.split if self.split else "",
            },
        }

    @override
    def transform_conversation(self, sample: pd.Series) -> Conversation:
        """Validate and transform the sample into Python `dict`."""
        sample_dict = self.transform(sample)
        prompt_message = sample_dict["prompt"][0]
        del sample_dict["prompt"]
        conversation_dict = {"messages": [prompt_message], "metadata": sample_dict}
        return Conversation.from_dict(conversation_dict)
