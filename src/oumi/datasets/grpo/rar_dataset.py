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

"""RaR (Rubrics as Rewards) dataset loaders.

This module provides dataset classes for loading the RaR-Medicine and RaR-Science
datasets from HuggingFace Hub. These datasets are from the paper:

"Rubrics as Rewards: Reinforcement Learning Beyond Verifiable Domains"
(arXiv:2507.17746)

The datasets contain prompts with structured rubric annotations that include:
- title: Short criterion name (2-4 words)
- description: Detailed description of the criterion
- weight: Importance weight (positive for Essential/Important/Optional, -ve for Pitfall)

Weight categories:
- Essential (weight=5): Core requirements for a correct answer
- Important (weight=3-4): Significant supporting points
- Optional (weight=1-2): Additional helpful information
- Pitfall (weight=-1 to -2): Common mistakes to avoid (negative criteria)
"""

from typing import Any

import pandas as pd
from typing_extensions import override

from oumi.core.datasets import BaseRubricDataset
from oumi.core.registry import register_dataset


@register_dataset("rar-medicine")
class RaRMedicineDataset(BaseRubricDataset):
    """Dataset for RaR-Medicine from the Rubrics as Rewards paper.

    This dataset contains 22.4k medical prompts with structured rubric annotations
    for training with GRPO. The prompts focus on complex medical reasoning tasks
    like diagnosis (50.3%) and treatment (16.0%).

    HuggingFace: https://huggingface.co/datasets/anisha2102/RaR-Medicine

    Example:
        >>> dataset = RaRMedicineDataset(split="train")
        >>> sample = dataset.raw(0)
        >>> print(sample["prompt"])
        >>> print(sample["rubrics"])  # List of weighted rubric dicts

    The rubrics follow this structure:
        {
            "name": "Identify Most Sensitive Modality",
            "description": "Essential Criteria: Identifies non-contrast helical CT...",
            "weight": 5,
            "evaluation_type": "binary"
        }
    """

    default_dataset = "anisha2102/RaR-Medicine"

    @override
    def transform(self, sample: pd.Series) -> dict[str, Any]:
        """Transform a sample into the format expected by GRPO trainer."""
        return {
            "prompt": sample["question"],
            "rubrics": [
                {
                    "name": r["title"],
                    "description": r["description"],
                    "weight": float(r["weight"]),
                }
                for r in sample["rubric"]
            ],
            "metadata": {
                "reference_answer": sample["reference_answer"],
                "question_source": sample["question_source"],
                "rubric_count": sample["rubric_count"],
            },
        }


@register_dataset("rar-science")
class RaRScienceDataset(RaRMedicineDataset):
    """Dataset for RaR-Science from the Rubrics as Rewards paper.

    This dataset contains 22.9k expert-level science prompts with structured
    rubric annotations for training with GRPO. The prompts are aligned with
    the GPQA Diamond benchmark, covering topics from quantum mechanics to
    molecular biology.

    HuggingFace: https://huggingface.co/datasets/anisha2102/RaR-Science

    Example:
        >>> dataset = RaRScienceDataset(split="train")
        >>> sample = dataset.raw(0)
        >>> print(sample["prompt"])
        >>> print(sample["rubrics"])  # List of weighted rubric dicts

    The rubrics follow this structure:
        {
            "name": "Temperature Conversion",
            "description": "Essential Criteria: The response must mention...",
            "weight": 5,
            "evaluation_type": "binary"
        }
    """

    default_dataset = "anisha2102/RaR-Science"
