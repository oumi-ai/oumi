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


@register_dataset("oumi-ai/oumi-letter-count")
class LetterCountGrpoDataset(BaseExperimentalGrpoDataset):
    """Dataset class for the `oumi-ai/oumi-letter-count` dataset.

    A sample from the dataset:
    {
        "prompt": "Can you let me know how many 'r's are in 'pandered'?",
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
