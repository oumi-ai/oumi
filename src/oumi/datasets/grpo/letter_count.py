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


@register_dataset("oumi-ai/oumi-letter-count")
class LetterCountGrpoDataset(BaseExperimentalGrpoDataset):
    """Dataset class for the `oumi-ai/oumi-letter-count` dataset.

    A sample from the dataset:
    {
        "conversation_id": "oumi_letter_count_0",
        "messages": [
            {
                "content": "Can you let me know how many 'r's are in 'pandered'?",
                "role": "user",
            }
        ],
        "metadata": {
            "letter": "r",
            "letter_count_integer": 1,
            "letter_count_string": "one",
            "unformatted_prompt": "Can you let me know how many {letter}s are in {word}?",
            "word": "pandered",
        },
    }
    """  # noqa: E501

    default_dataset = "oumi-ai/oumi-letter-count"

    @override
    def transform(self, sample: pd.Series) -> dict:
        """Validate and transform the sample into Python `dict`."""
        return {
            "prompt": sample["messages"],
            "letter_count": sample["metadata"]["letter_count_integer"],
        }

    @override
    def transform_conversation(self, example: pd.Series) -> Conversation:
        """Preprocesses the inputs of the example and returns a dictionary.

        Args:
            example (dict): The example containing the input and instruction.

        Returns:
            dict: The preprocessed inputs as a dictionary.

        """
        example_dict = example.to_dict()
        # Convert messages from np.ndarray to list.
        example_dict["messages"] = example_dict["messages"].tolist()
        return Conversation.from_dict(example_dict)
