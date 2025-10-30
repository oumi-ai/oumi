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

"""Dataset for knowledge distillation training."""

from typing import Any, Optional, Union

import pandas as pd

from oumi.core.datasets import BaseMapDataset
from oumi.core.registry import register_dataset


@register_dataset("distillation")
class DistillationDataset(BaseMapDataset):
    """Dataset for GKD training that provides prompts in TRL's conversational format.

    This dataset returns untokenized conversations with user prompts and optional
    assistant responses. TRL's GKDTrainer will handle tokenization and on-policy
    distillation during training.

    Expected input format:
        - prompt_column: str - The input prompt text
        - response_column: str (optional) - The reference response text

    Output format (TRL-compatible):
        - messages: List of message dicts with 'role' and 'content' keys

    Note:
        When response_column is not provided or empty, an empty placeholder assistant
        message is used. This is required because TRL's collator uses messages[:-1]
        to extract the prompt.
    """

    def __init__(
        self,
        *,
        prompt_column: str = "prompts",
        response_column: Optional[str] = None,
        tokenizer=None,  # Accept but ignore tokenizer for compatibility
        **kwargs,
    ) -> None:
        """Initialize the distillation dataset.

        Args:
            prompt_column: Name of the column containing prompts (default: "prompts").
            response_column: Name of the column containing responses (optional).
                If provided, uses actual responses; otherwise uses empty placeholder.
            tokenizer: Ignored - kept for compatibility with other datasets.
            **kwargs: Additional arguments passed to BaseMapDataset.
        """
        self.prompt_column = prompt_column
        self.response_column = response_column
        super().__init__(**kwargs)

        self._data = self._load_data()

    def transform(self, example: Union[dict, pd.Series]) -> dict[str, Any]:
        """Transform the example into TRL's expected format.

        Args:
            example: Raw example from the dataset.

        Returns:
            Dict with 'messages' field in TRL format.
        """
        prompt_text = example[self.prompt_column]

        # Get response if available, otherwise use empty placeholder
        if self.response_column and self.response_column in example:
            response_text = str(example[self.response_column])
        else:
            response_text = ""

        # For GKD, we need full conversations with user + assistant messages
        # TRL's collator uses messages[:-1] to extract the prompt, so we need >= 2 messages
        return {
            "messages": [
                {"role": "user", "content": prompt_text},
                {"role": "assistant", "content": response_text},
            ]
        }
