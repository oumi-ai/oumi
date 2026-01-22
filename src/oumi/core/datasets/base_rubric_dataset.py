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

"""Base class for rubric-based datasets."""

from abc import abstractmethod
from typing import Any, TypedDict

import pandas as pd

from oumi.core.datasets.base_map_dataset import BaseMapDataset


class Rubric(TypedDict):
    """A single rubric criterion for evaluating responses."""

    name: str
    """Short identifier for the criterion (e.g., 'Correct Diagnosis')."""

    description: str
    """Detailed description of what the criterion evaluates."""

    weight: float
    """Importance weight. Positive for desired criteria, negative for pitfalls."""


class BaseRubricDataset(BaseMapDataset):
    """Base class for rubric-based datasets.

    This provides common functionality for datasets used with rubric-based
    reward functions in GRPO training. Subclasses should implement the
    `transform` method to return the expected format.

    Expected transform() output format:
        {
            "prompt": str,                # The user prompt/question
            "rubrics": list[Rubric],      # List of evaluation criteria
            "system_prompt": str | None,  # Optional system prompt
            "metadata": dict | None,      # Optional dataset-specific metadata
        }
    """

    def __init__(
        self,
        *,
        dataset_name: str | None = None,
        dataset_path: str | None = None,
        split: str | None = None,
        **kwargs,
    ) -> None:
        """Initializes the BaseRubricDataset."""
        super().__init__(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            split=split,
            **kwargs,
        )
        self._data = self._load_data()

    @abstractmethod
    def transform(self, sample: pd.Series) -> dict[str, Any]:
        """Transform a raw sample into the standard rubric format.

        Subclasses must override this method to return:
            {
                "prompt": str,
                "rubrics": list[Rubric],
                "system_prompt": str | None,  # optional
                "metadata": dict | None,      # optional
            }
        """
