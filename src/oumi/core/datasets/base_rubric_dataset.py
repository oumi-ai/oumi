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

from typing import Any

import pandas as pd

from oumi.core.datasets.base_map_dataset import BaseMapDataset
from oumi.core.types.conversation import Conversation


class BaseRubricDataset(BaseMapDataset):
    """Base class for rubric-based datasets.

    This provides common functionality for datasets used with rubric-based
    reward functions in GRPO training. Subclasses should implement the
    `transform` method to extract prompt, rubrics, and optional system_prompt.
    """

    def __init__(
        self,
        *,
        dataset_name: str | None = None,
        dataset_path: str | None = None,
        split: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            split=split,
            **kwargs,
        )
        self._data = self._load_data()

    def conversation(self, idx: int) -> Conversation:
        """Returns the conversation at the specified index."""
        sample = self.raw(idx)
        return self.transform_conversation(sample)

    def conversations(self) -> list[Conversation]:
        """Returns a list of all conversations."""
        return [self.conversation(i) for i in range(len(self))]

    def transform_conversation(self, sample: dict | pd.Series) -> Conversation:
        """Converts the input sample to a Conversation."""
        if isinstance(sample, dict):
            sample = pd.Series(sample)
        transformed = self.transform(sample)

        messages = []
        if "system_prompt" in transformed and transformed["system_prompt"]:
            messages.append(
                {
                    "content": transformed["system_prompt"],
                    "role": "system",
                }
            )
        messages.append(
            {
                "content": transformed["prompt"],
                "role": "user",
            }
        )

        return Conversation.from_dict({"messages": messages})

    def transform(self, sample: pd.Series) -> dict[str, Any]:
        """Transform the sample. Subclasses should override this method."""
        raise NotImplementedError("Subclasses must implement transform()")
