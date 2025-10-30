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

    This dataset returns untokenized conversations with only user messages (prompts).
    TRL's GKDTrainer will handle tokenization and generate completions during training.

    Expected input format:
        - prompts: str - The input prompt text

    Output format (TRL-compatible):
        - messages: List of message dicts with 'role' and 'content' keys
    """

    def __init__(
        self,
        *,
        prompts_col: str = "prompts",
        tokenizer=None,  # Accept but ignore tokenizer for compatibility
        **kwargs,
    ) -> None:
        """Initialize the distillation dataset.

        Args:
            prompts_col: Name of the column containing prompts (default: "prompts").
            tokenizer: Ignored - kept for compatibility with other datasets.
            **kwargs: Additional arguments passed to BaseMapDataset.
        """
        self.prompts_col = prompts_col
        super().__init__(**kwargs)
        self._load_data()

    def _load_data(self):
        """Load the dataset from the specified source."""
        import datasets

        # Load the dataset
        if self.dataset_path and self.dataset_path.endswith(('.json', '.jsonl')):
            # Load from local JSON/JSONL file
            self._data = pd.read_json(self.dataset_path, lines=True)
        elif self.dataset_path:
            # Try loading as HuggingFace dataset from disk
            hf_dataset = datasets.load_from_disk(self.dataset_path)
            self._data = hf_dataset.to_pandas()
        else:
            # Load from HuggingFace hub
            hf_dataset = datasets.load_dataset(
                self.dataset_name,
                name=self.dataset_subset,
                split=self.split,
                trust_remote_code=self.trust_remote_code,
            )
            self._data = hf_dataset.to_pandas()

        # Apply split if specified
        if self.split and '[' in self.split:
            # Handle slice notation like "train[:50]"
            split_name = self.split.split('[')[0]
            slice_str = self.split.split('[')[1].rstrip(']')

            if ':' in slice_str:
                parts = slice_str.split(':')
                start = int(parts[0]) if parts[0] else None
                end = int(parts[1]) if parts[1] else None
                self._data = self._data.iloc[start:end]
            else:
                # Single index
                idx = int(slice_str)
                self._data = self._data.iloc[:idx]

    def transform(self, example: Union[dict, pd.Series]) -> dict[str, Any]:
        """Transform the example into TRL's expected format.

        Args:
            example: Raw example from the dataset.

        Returns:
            Dict with 'messages' field in TRL format.
        """
        prompt_text = example[self.prompts_col]

        # For GKD, we need full conversations with user + assistant messages
        # TRL's collator uses messages[:-1] to extract the prompt, so we need >= 2 messages
        # The assistant response can be a placeholder - GKD will use teacher/student distribution
        return {
            "messages": [
                {"role": "user", "content": prompt_text},
                {"role": "assistant", "content": ""}  # Empty placeholder for GKD
            ]
        }
