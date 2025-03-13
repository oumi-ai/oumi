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

from oumi.core.datasets.base_map_dataset import BaseMapDataset
from oumi.core.tokenizers.base_tokenizer import BaseTokenizer

_PROMPT_KEY = "prompt"
_RESPONSE_KEY = "response"
_LABEL_KEY = "label"  # True for desirable, False for undesirable

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
            A dictionary with the following keys:
            - prompt: The input prompt
            - response: The model's response
            - label: Boolean indicating if the response is desirable (True) or undesirable (False)
        """
        prompt = sample[_PROMPT_KEY]
        response = sample[_RESPONSE_KEY]
        label = sample[_LABEL_KEY]

        return {
            _PROMPT_KEY: prompt,
            _RESPONSE_KEY: response,
            _LABEL_KEY: label,
        }

    def transform(self, sample: dict) -> dict:
        """Transform the sample to the KTO format."""
        return self.transform_kto(sample) 