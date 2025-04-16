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

import pandas as pd
from typing_extensions import override

from oumi.core.datasets.base_grpo_dataset import BaseExperimentalGrpoDataset
from oumi.core.registry import register_dataset
from oumi.core.types.conversation import Conversation

_SYSTEM_PROMPT = (
    "Your final answer should be formatted as "
    r'```json YOUR_ANSWER```. For example, if the answer is {"a": 1}, '
    r'you should output ```json {"a": 1}}```.'
)


@register_dataset("oumi-ai/berrybench-v0.1.1")
class BerryBenchGrpoDataset(BaseExperimentalGrpoDataset):
    r"""Dataset class for the `oumi-ai/berrybench-v0.1.1` dataset.

    A sample from the dataset:
    {
        "messages": [
            {
                "content": "Return a JSON object showing the frequency of each character in the word '黒い'. Only include characters that appear in the word.",
                "role": "user",
            }
        ],
        "metadata": {
            "character_count": 2,
            "difficulty": 3,
            "expected_response": '{"\\u9ed2": 1, "\\u3044": 1}',
            "language": "japanese",
            "word": "黒い",
        },
    }
    """  # noqa: E501

    default_dataset = "oumi-ai/berrybench-v0.1.1"

    @override
    def transform(self, sample: pd.Series) -> dict:
        """Transform the sample into Python `dict`."""
        # Add system prompt before user prompt.
        system_message = {"content": _SYSTEM_PROMPT, "role": "system"}
        messages = [system_message, sample["messages"][0]]
        return {
            "prompt": messages,
            "metadata": sample["metadata"],
        }

    @override
    def transform_conversation(self, sample: pd.Series) -> Conversation:
        """Converts the input sample to a Conversation.

        Args:
            sample (dict): The input example.

        Returns:
            Conversation: The resulting conversation.

        """
        # Example is already in conversation format and only needs light processing.
        sample_dict = sample.to_dict()
        # Add system prompt before user prompt.
        system_message = {"content": _SYSTEM_PROMPT, "role": "system"}
        messages = [system_message, sample["messages"][0]]
        sample_dict["messages"] = messages
        return Conversation.from_dict(sample_dict)
