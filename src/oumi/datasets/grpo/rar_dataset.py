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

from oumi.core.datasets.base_map_dataset import BaseMapDataset
from oumi.core.registry import register_dataset
from oumi.core.types.conversation import Conversation

# Weight mapping from categorical importance levels to numeric weights
# Based on the RaR paper's default scheme
IMPORTANCE_WEIGHTS = {
    "Essential": 5,
    "Important": 4,
    "Optional": 2,
    "Pitfall": -1,  # Negative weight for anti-patterns
}


def _infer_importance_level(weight: int) -> str:
    """Infer importance level from numeric weight.

    Args:
        weight: The numeric weight value.

    Returns:
        The importance level string.
    """
    if weight >= 5:
        return "Essential"
    elif weight >= 3:
        return "Important"
    elif weight >= 1:
        return "Optional"
    else:
        return "Pitfall"


def normalize_rar_rubrics(rubrics_raw: list) -> list[dict[str, Any]]:
    """Normalize RaR rubrics to weighted format.

    Converts from RaR format:
        {"title": "...", "description": "...", "weight": 5}

    To our format:
        {"name": "...", "description": "...", "weight": 5,
         "importance_level": "Essential", "evaluation_type": "binary"}

    Args:
        rubrics_raw: List of rubric dicts from the RaR dataset.

    Returns:
        List of normalized rubric dicts.
    """
    normalized = []
    for i, rubric in enumerate(rubrics_raw):
        if isinstance(rubric, dict):
            weight = int(rubric.get("weight", 1))
            normalized.append(
                {
                    "name": rubric.get("title", f"rubric_{i + 1}"),
                    "description": rubric.get("description", ""),
                    "weight": weight,
                    "importance_level": _infer_importance_level(weight),
                    "evaluation_type": "binary",
                }
            )
        elif isinstance(rubric, str):
            # Handle string format (from rubric_list)
            normalized.append(
                {
                    "name": f"rubric_{i + 1}",
                    "description": rubric,
                    "weight": 1,
                    "importance_level": "Optional",
                    "evaluation_type": "binary",
                }
            )
    return normalized


class _RaRBaseDataset(BaseMapDataset):
    """Base class for RaR (Rubrics as Rewards) datasets.

    Provides shared implementation for loading RaR datasets from HuggingFace Hub.
    Subclasses only need to set `default_dataset` and register themselves.
    """

    def __init__(
        self,
        *,
        dataset_name: str | None = None,
        dataset_path: str | None = None,
        split: str | None = "train",
        **kwargs,
    ) -> None:
        """Initialize the RaR dataset.

        Args:
            dataset_name: Name of the dataset.
            dataset_path: Optional local path to the dataset.
            split: Dataset split to load ("train", "val", or "test").
            **kwargs: Additional arguments passed to BaseMapDataset.
        """
        super().__init__(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            split=split,
            **kwargs,
        )
        self._data = self._load_data()

    @override
    def transform(self, sample: pd.Series) -> dict[str, Any]:
        """Transform a sample into the format expected by GRPO trainer.

        Maps the RaR dataset format to our rubric format:
        - question -> prompt
        - rubric -> rubrics (normalized to our weighted format)
        - reference_answer -> reference_answer (optional, for evaluation)

        Args:
            sample: A pandas Series containing the raw data.

        Returns:
            A dict with 'prompt', 'rubrics', and optional fields.
        """
        prompt = sample.get("question", "")
        rubrics_raw = sample.get("rubric", [])
        reference_answer = sample.get("reference_answer", None)
        question_source = sample.get("question_source", None)
        rubric_count = sample.get("rubric_count", None)

        # Normalize rubrics to our format (handle None case)
        rubrics = normalize_rar_rubrics(rubrics_raw if rubrics_raw is not None else [])

        result = {
            "prompt": str(prompt).strip(),
            "rubrics": rubrics,
        }

        # Add optional fields if present
        if reference_answer:
            result["reference_answer"] = str(reference_answer).strip()
        if question_source:
            result["question_source"] = str(question_source)
        if rubric_count is not None:
            result["rubric_count"] = int(rubric_count)

        return result

    def conversation(self, idx: int) -> Conversation:
        """Returns the conversation at the specified index.

        Args:
            idx: The index of the conversation to retrieve.

        Returns:
            The conversation at the specified index.
        """
        sample = self.raw(idx)
        return self.transform_conversation(sample)

    def conversations(self) -> list[Conversation]:
        """Returns a list of all conversations."""
        return [self.conversation(i) for i in range(len(self))]

    def transform_conversation(self, sample: dict | pd.Series) -> Conversation:
        """Converts the input sample to a Conversation.

        Args:
            sample: The input example.

        Returns:
            The resulting conversation.
        """
        # Convert dict to Series if needed for transform()
        if isinstance(sample, dict):
            sample = pd.Series(sample)
        transformed = self.transform(sample)
        messages = [
            {
                "content": transformed["prompt"],
                "role": "user",
            }
        ]
        conversation_dict = {"messages": messages}
        return Conversation.from_dict(conversation_dict)


@register_dataset("rar-medicine")
class RaRMedicineDataset(_RaRBaseDataset):
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
            "importance_level": "Essential",
            "evaluation_type": "binary"
        }
    """

    default_dataset = "anisha2102/RaR-Medicine"


@register_dataset("rar-science")
class RaRScienceDataset(_RaRBaseDataset):
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
            "importance_level": "Essential",
            "evaluation_type": "binary"
        }
    """

    default_dataset = "anisha2102/RaR-Science"
