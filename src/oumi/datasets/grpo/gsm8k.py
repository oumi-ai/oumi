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

"""GSM8K dataset for GRPO training."""

import re

import pandas as pd
from typing_extensions import override

from oumi.core.datasets.base_grpo_dataset import BaseExperimentalGrpoDataset
from oumi.core.registry import register_dataset


@register_dataset("openai/gsm8k")
class GSM8KGrpoDataset(BaseExperimentalGrpoDataset):
    """Dataset class for the `openai/gsm8k` mathematical reasoning dataset.

    This dataset adapts the GSM8K dataset for reinforcement learning with GRPO.
    GSM8K contains math word problems requiring multi-step reasoning.
    """

    default_dataset = "openai/gsm8k"
    default_subset = "main"

    @staticmethod
    def extract_answer(solution_str: str) -> str:
        """Extract the numerical answer from the solution string.

        Args:
            solution_str: The solution string from GSM8K dataset

        Returns:
            The extracted numerical answer

        Raises:
            ValueError: If no answer can be extracted
        """
        match = re.search(r"#### (\-?[0-9\.\,]+)", solution_str)
        if match is None:
            raise ValueError(f"Could not extract solution from: {solution_str}")
        return match.group(1).replace(",", "")

    @override
    def transform(self, sample: pd.Series) -> dict:
        """Transform GSM8K sample to GRPO format.

        Args:
            sample: Raw sample from the GSM8K dataset

        Returns:
            Transformed sample with prompt and completion fields
        """
        # Add a standard instruction to encourage step-by-step reasoning
        instruction = (
            "Let's solve this step by step and provide the final answer after ####."
        )
        question = sample["question"] + "\n\n" + instruction
        answer = sample["answer"]

        try:
            # Extract the numerical answer for potential reward calculation
            extracted_answer = self.extract_answer(answer)  # type: ignore
        except ValueError:
            # Use empty string if extraction fails
            extracted_answer = ""

        return {
            "prompt": question,
            "completion": answer,
            "metadata": {
                "ground_truth": extracted_answer,
                "raw_question": sample["question"],
                "raw_answer": answer,
            },
        }
