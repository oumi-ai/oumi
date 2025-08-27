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

from typing import Optional, Union, cast

import numpy as np
from typing_extensions import override

from oumi.core.datasets.base_map_dataset import BaseMapDataset
from oumi.core.tokenizers.base_tokenizer import BaseTokenizer
from oumi.core.types.conversation import Role

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

            OR

            {
                "prompt": "How is the weather in Tokyo?",
                "chosen": "preferred response",
                "rejected": "rejected response"
            }
        """
        super().__init__(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            split=split,
            **kwargs,
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
            _PROMPT_KEY: self._to_messages_list(prompt, Role.USER),
            _CHOSEN_KEY: self._to_messages_list(chosen_chat, Role.ASSISTANT),
            _REJECTED_KEY: self._to_messages_list(rejected_chat, Role.ASSISTANT),
        }

    def _process_sample(
        self,
        features,
    ):
        """Tokenize a row of the dataset.

        Example:
        ```python
        >>> from transformers import GPT2Tokenizer
        >>> tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        >>> features = {"prompt": "The sky is", "chosen": " blue", "rejected": " green"}
        >>> DPOTrainer.tokenize_row(
        ...     features, tokenizer, max_prompt_length=3, max_completion_length=3,
        ...     add_special_tokens=False,
        ... )
        {'prompt_input_ids': [464, 6766, 318], 'chosen_input_ids': [4171, 50256],
        'rejected_input_ids': [4077, 50256]}
        ```
        """
        if self._tokenizer is None:
            raise ValueError("Tokenizer is required to process a sample.")

        # Apply the chat template to the prompt.
        prompt = self._tokenizer.apply_chat_template(features["prompt"], tokenize=False)
        prompt = cast(str, prompt)

        # Apply the chat template to the chosen and rejected turns.
        # To get only the completion part, we tokenizer the prompt + chosen/rejected
        # and then remove the prompt prefix.
        prompt_chosen = self._tokenizer.apply_chat_template(
            features["prompt"] + features["chosen"], tokenize=False
        )
        prompt_chosen = cast(str, prompt_chosen)
        chosen = prompt_chosen[len(prompt) :]

        prompt_rejected = self._tokenizer.apply_chat_template(
            features["prompt"] + features["rejected"], tokenize=False
        )
        prompt_rejected = cast(str, prompt_rejected)
        rejected = prompt_rejected[len(prompt) :]

        # Tokenize the prompt, chosen, and rejected turns.
        prompt_input_ids = self._tokenizer(prompt, add_special_tokens=False)[
            "input_ids"
        ]
        chosen_input_ids = self._tokenizer(chosen, add_special_tokens=False)[
            "input_ids"
        ]
        chosen_input_ids = cast(list[int], chosen_input_ids)
        rejected_input_ids = self._tokenizer(rejected, add_special_tokens=False)[
            "input_ids"
        ]
        rejected_input_ids = cast(list[int], rejected_input_ids)
        chosen_input_ids = chosen_input_ids + [self._tokenizer.eos_token_id]
        rejected_input_ids = rejected_input_ids + [self._tokenizer.eos_token_id]

        return {
            "prompt_input_ids": prompt_input_ids,
            "chosen_input_ids": chosen_input_ids,
            "rejected_input_ids": rejected_input_ids,
        }

    @override
    def transform(self, sample: dict) -> dict:
        """Transform the samples to the Oumi format."""
        return self._process_sample(self.transform_preference(sample))

    def _to_messages_list(
        self, turn: Union[str, dict, list[dict]], role: Role
    ) -> list[dict]:
        """Convert a turn to a conversation dictionary."""
        if isinstance(turn, str):
            return [{"role": role.value, "content": turn}]
        if isinstance(turn, dict):
            return [turn]
        if isinstance(turn, np.ndarray):
            return list(turn)
        if isinstance(turn, list):
            return turn

        raise ValueError(f"Invalid turn type: {type(turn)}")


class BaseExperimentalDpoDataset(BaseDpoDataset):
    """Preprocess the samples to the Oumi format.

    Warning:
        This class is experimental and subject to change.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initializes a new instance of the BaseExperimentalDpoDataset class."""
        from oumi.utils.logging import logger

        logger.warning(
            "`BaseExperimentalDpoDataset` is deprecated and will be removed in the "
            "future. Please use `BaseDpoDataset` instead."
        )

        super().__init__(*args, **kwargs)
