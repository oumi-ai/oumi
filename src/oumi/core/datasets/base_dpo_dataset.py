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

from typing import Optional

from typing_extensions import override

from oumi.core.datasets.base_map_dataset import BaseMapDataset
from oumi.core.tokenizers.base_tokenizer import BaseTokenizer

_PROMPT_KEY = "prompt"
_CHOSEN_KEY = "chosen"
_REJECTED_KEY = "rejected"

_ROLE = "role"
_CONTENT = "content"
_ASSISTANT = "assistant"


class BaseDpoDataset(BaseMapDataset):
    """Preprocess the samples to the Oumi format."""

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
        """Initializes a new instance of the BaseDpoDataset class.

        The dataset expects data in the format::

            {
                "prompt": "How is the weather in Tokyo?",
                "chosen": [{"role": "assistant", "content": "It's sunny and warm."}],
                "rejected": [{"role": "assistant", "content": "It's rainy and cold."}]
            }
        """
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

    def transform_preference(self, samples: dict) -> dict:
        """Transform the samples to the Oumi format."""
        prompt = samples[_PROMPT_KEY]
        chosen_chat = samples[_CHOSEN_KEY]
        rejected_chat = samples[_REJECTED_KEY]

        return {
            _PROMPT_KEY: prompt,
            _CHOSEN_KEY: chosen_chat,
            _REJECTED_KEY: rejected_chat,
        }

    @override
    def transform(self, sample: dict) -> dict:
        """Transform the samples to the Oumi format."""
        return self.transform_preference(sample)
