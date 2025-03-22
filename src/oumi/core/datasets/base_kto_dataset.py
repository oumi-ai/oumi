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

"""Base dataset class for KTO (Kahneman-Tversky Optimization).

This module provides a base class for datasets used in KTO training.
Unlike DPO which requires preference pairs, KTO works with simple binary feedback
indicating whether an output is desirable or undesirable.
"""

from typing import Optional

import datasets
from typing_extensions import override

from oumi.core.datasets.base_map_dataset import BaseMapDataset
from oumi.core.tokenizers.base_tokenizer import BaseTokenizer

_PROMPT_KEY = "prompt"
_COMPLETION_KEY = "completion"
_LABEL_KEY = "label"  # True for desirable, False for undesirable

_ROLE = "role"
_CONTENT = "content"
_ASSISTANT = "assistant"


class BaseKtoDataset(BaseMapDataset):
    """Base class for KTO datasets.

    This class provides a foundation for creating KTO datasets that work with
    binary feedback (desirable/undesirable) rather than preference pairs.

    Warning:
        This class is experimental and subject to change.
    """

    def __init__(
        self,
        *,
        dataset_name: Optional[str] = None,
        dataset_path: Optional[str] = None,
        split: Optional[str] = None,
        tokenizer: Optional[BaseTokenizer] = None,
        return_tensors: bool = False,
        **kwargs,
    ) -> None:
        """Initializes a new instance of the BaseKtoDataset class."""
        super().__init__(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            split=split,
            **kwargs,
        )

        if return_tensors:
            raise NotImplementedError(
                "return_tensors=True is not implemented for this class"
            )

        self._tokenizer = tokenizer
        self._return_tensors = return_tensors

        self._data = self._load_data()

    def transform_kto(self, sample: dict) -> dict:
        """Transform the sample to the KTO format.

        Args:
            sample: A dictionary containing the raw sample data.

        Returns:
            A dictionary with the basic format expected by TRL:
            - prompt: The input prompt
            - completion: The model's response
            - label: Boolean indicating if the response is desirable or undesirable
        """
        prompt = sample[_PROMPT_KEY]
        completion = sample[_COMPLETION_KEY]
        label = sample[_LABEL_KEY]

        # Extract text from completion if it's in chat format
        if isinstance(completion, list):
            completion = self._extract_from_chat_format(completion)

        return {
            _PROMPT_KEY: prompt,
            _COMPLETION_KEY: completion,
            _LABEL_KEY: label,
        }

    @override
    def transform(self, sample: dict) -> dict:
        """Transform the sample to the KTO format."""
        return self.transform_kto(sample)

    def _extract_from_chat_format(self, sample) -> str:
        """Extract the last 'assistant' turn in the chat."""
        if not isinstance(sample, list):
            return sample

        for turn in sample[::-1]:
            if turn[_ROLE] == _ASSISTANT:
                return turn[_CONTENT]

        raise ValueError("No chat turn was found with an 'assistant' role.")

    @property
    def _kto_features(self) -> datasets.Features:
        """Get the explicit feature schema required for KTO training."""
        return datasets.Features(
            {
                "prompt": datasets.Value("string"),
                "completion": datasets.Value("string"),
                "label": datasets.Value("bool"),
            }
        )

    def _detect_features_and_estimate_element_size_bytes(self, generator):
        """Override to use explicit KTO features."""
        from oumi.core.datasets.base_map_dataset import _InferredFeatureMap
        from oumi.utils.torch_utils import estimate_sample_dict_size_in_bytes

        # Collect a few samples to estimate average size
        samples = []
        for _ in range(min(10, len(self))):  # Use up to 10 samples
            try:
                samples.append(next(generator))
            except StopIteration:
                break

        # Calculate estimated element size based on actual samples
        element_size = 1024  # Default fallback
        if samples:
            # Get average size of samples
            element_size = sum(
                estimate_sample_dict_size_in_bytes(s) for s in samples
            ) // len(samples)
            # Add 20% buffer for safety
            element_size = int(element_size * 1.2)

        # Return features optimized for KTO training with proper size estimate
        return _InferredFeatureMap(
            feature_map=self._kto_features,
            is_feature_map_optimized=True,
            element_size_in_bytes=element_size,
            multimodal=False,
        )
