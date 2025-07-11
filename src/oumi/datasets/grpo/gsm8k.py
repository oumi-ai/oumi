# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

"""Derived from https://github.com/volcengine/verl/blob/main/examples/data_preprocess/gsm8k.py.

This file was slightly modified to inherit from the `BaseExperimentalGrpoDataset` class.
"""

import re

import pandas as pd
from typing_extensions import override

from oumi.core.datasets.base_grpo_dataset import BaseExperimentalGrpoDataset
from oumi.core.registry import register_dataset
from oumi.core.types.conversation import Conversation


def _extract_solution(solution_str):
    """Extract the solution from the response."""
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


@register_dataset("openai/gsm8k")
class Gsm8kGrpoDataset(BaseExperimentalGrpoDataset):
    """Dataset class for the `openai/gsm8k` dataset.

    A sample from the dataset:
    {
        "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "answer": "Natalia sold 48/2 = <<48/2=24>>24 clips in May.
                   Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.
                   #### 72"
    }
    """  # noqa: E501

    default_dataset = "openai/gsm8k"

    @override
    def transform(self, sample: pd.Series) -> dict:
        """Validate and transform the sample into Python `dict`."""
        instruction = (
            'Let\'s think step by step and output the final answer after "####".'
        )
        prompt = f"{sample['question']} {instruction}"
        solution = _extract_solution(sample["answer"])
        return {
            "data_source": "gsm8k",
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
                "answer": sample["answer"],
                "question": sample["question"],
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
