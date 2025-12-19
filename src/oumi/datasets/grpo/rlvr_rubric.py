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

"""RLVR (RL from Verifiable Rewards) dataset with rubric-based evaluation.

This module provides a dataset class for loading prompts with associated rubrics
for use with the rubric-based reward function in GRPO training.

Supports two rubric formats:

1. Simple format (list of strings):
   ```json
   {
       "prompt": "Write a product description.",
       "rubrics": ["Mentions features", "Professional tone"]
   }
   ```

2. Weighted format (list of rubric objects):
   ```json
   {
       "prompt": "Handle this customer complaint.",
       "system_prompt": "You are a customer service agent.",
       "rubrics": [
           {"name": "empathy", "description": "Shows empathy", "weight": 2.0},
           {"name": "solution", "description": "Offers solution", "weight": 1.5}
       ],
       "metadata": {"category": "support", "complexity": "medium"}
   }
   ```
"""

import json
from typing import Any, Optional, Union

import pandas as pd
from typing_extensions import override

from oumi.core.datasets.base_map_dataset import BaseMapDataset
from oumi.core.registry import register_dataset
from oumi.core.types.conversation import Conversation


@register_dataset("oumi-rlvr-rubric")
class RlvrRubricDataset(BaseMapDataset):
    """Dataset for RLVR training with rubric-based rewards.

    This dataset loads prompts with associated rubrics from a JSONL file.
    Supports both simple string rubrics and weighted rubric objects.

    Required fields:
    - prompt: The task/instruction

    Optional fields:
    - rubrics: List of rubric strings OR rubric objects with:
        - name: Short identifier for the rubric
        - description: Full description of the criterion
        - weight: Numeric weight (default: 1.0)
        - evaluation_type: "binary" or "graded" (default: "binary")
    - system_prompt: System instruction for the model
    - prompt_id: Unique identifier for the prompt
    - metadata: Dict with additional info (category, sentiment, etc.)

    Example weighted rubric format:
    ```json
    {
        "prompt_id": "cx-001",
        "prompt": "Customer says: I need help with my order.",
        "system_prompt": "You are a helpful customer service agent.",
        "rubrics": [
            {
                "name": "empathy",
                "description": "Response shows empathy for the customer",
                "weight": 2.0,
                "evaluation_type": "binary"
            }
        ],
        "metadata": {"category": "support"}
    }
    ```
    """

    default_dataset = "oumi-rlvr-rubric"

    def __init__(
        self,
        *,
        dataset_name: Optional[str] = None,
        dataset_path: Optional[str] = None,
        split: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Initialize the RLVR Rubric dataset.

        Args:
            dataset_name: Name of the dataset (optional).
            dataset_path: Path to the JSONL file containing the data.
            split: Dataset split (optional, not used for local files).
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

        Args:
            sample: A pandas Series containing the raw data.

        Returns:
            A dict with 'prompt', 'rubrics', and optional fields.
        """
        prompt = sample.get("prompt", "")
        rubrics = sample.get("rubrics", [])
        system_prompt = sample.get("system_prompt", None)
        prompt_id = sample.get("prompt_id", None)
        metadata = sample.get("metadata", None)

        # Handle rubrics that might be stored as JSON string
        if isinstance(rubrics, str):
            try:
                rubrics = json.loads(rubrics)
            except json.JSONDecodeError:
                rubrics = [rubrics]

        # Ensure rubrics is a list
        if not isinstance(rubrics, list):
            rubrics = [str(rubrics)]

        # Normalize rubrics to the weighted format
        normalized_rubrics = self._normalize_rubrics(rubrics)

        # Handle metadata that might be stored as JSON string
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = None

        result = {
            "prompt": str(prompt).strip(),
            "rubrics": normalized_rubrics,
        }

        # Add optional fields if present
        if system_prompt:
            result["system_prompt"] = str(system_prompt).strip()
        if prompt_id:
            result["prompt_id"] = str(prompt_id)
        if metadata:
            result["metadata"] = metadata

        return result

    def _normalize_rubrics(self, rubrics: list) -> list[dict[str, Any]]:
        """Normalize rubrics to the weighted format.

        Converts simple string rubrics to the full dict format with default weights.

        Args:
            rubrics: List of strings or dicts.

        Returns:
            List of normalized rubric dicts.
        """
        normalized = []
        for i, rubric in enumerate(rubrics):
            if isinstance(rubric, str):
                # Simple string format -> convert to weighted format
                normalized.append(
                    {
                        "name": f"rubric_{i + 1}",
                        "description": rubric,
                        "weight": 1.0,
                        "evaluation_type": "binary",
                    }
                )
            elif isinstance(rubric, dict):
                # Already in dict format, ensure required fields
                normalized.append(
                    {
                        "name": rubric.get("name", f"rubric_{i + 1}"),
                        "description": rubric.get("description", str(rubric)),
                        "weight": float(rubric.get("weight", 1.0)),
                        "evaluation_type": rubric.get("evaluation_type", "binary"),
                    }
                )
            else:
                # Unknown format, convert to string
                normalized.append(
                    {
                        "name": f"rubric_{i + 1}",
                        "description": str(rubric),
                        "weight": 1.0,
                        "evaluation_type": "binary",
                    }
                )
        return normalized

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

    def transform_conversation(self, sample: Union[dict, pd.Series]) -> Conversation:
        """Converts the input sample to a Conversation.

        Args:
            sample: The input example.

        Returns:
            The resulting conversation.
        """
        transformed = self.transform(sample)
        messages = []

        # Add system message if present
        if "system_prompt" in transformed:
            messages.append(
                {
                    "content": transformed["system_prompt"],
                    "role": "system",
                }
            )

        # Add user message
        messages.append(
            {
                "content": transformed["prompt"],
                "role": "user",
            }
        )

        conversation_dict = {"messages": messages}
        return Conversation.from_dict(conversation_dict)


@register_dataset("oumi-rlvr-rubric-hf")
class RlvrRubricHfDataset(RlvrRubricDataset):
    """RLVR Rubric dataset that can load from HuggingFace Hub.

    This is a convenience class for loading RLVR datasets from
    HuggingFace Hub. The dataset should have 'prompt' and 'rubrics' columns.
    """

    pass
