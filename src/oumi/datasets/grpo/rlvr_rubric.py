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

"""RLVR (RL from Verifiable Rewards) dataset with rubric-based evaluation."""

from typing import Any

import pandas as pd
from typing_extensions import override

from oumi.core.datasets import BaseRubricDataset
from oumi.core.registry import register_dataset


@register_dataset("oumi-rlvr-rubric")
class RlvrRubricDataset(BaseRubricDataset):
    """Dataset for RLVR training with rubric-based rewards."""

    default_dataset = "oumi-rlvr-rubric"

    @override
    def transform(self, sample: pd.Series) -> dict[str, Any]:
        """Transform the sample into the format expected by GRPO trainer.

        Expects well-formed input with:
        - prompt: str
        - rubrics: list of dicts with name, description, weight, evaluation_type
        - system_prompt: optional str
        """
        result: dict[str, Any] = {
            "prompt": sample["prompt"],
            "rubrics": sample["rubrics"],
        }

        if system_prompt := sample.get("system_prompt"):
            result["system_prompt"] = system_prompt

        return result
