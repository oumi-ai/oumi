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

from typing import Any

import trl

from oumi.core.tokenizers.base_tokenizer import BaseTokenizer
from oumi.builders.collators import log_tokenized_example

_INPUT_IDS_KEY = "input_ids"


class TextCompletionsCollatorWithPadding:
    def __init__(
        self, tokenizer: BaseTokenizer, instruction_prefix: str, response_prefix: str, debug: bool = False
    ):
        """Custom collator for text LLM training.

        Args:
        tokenizer: The tokenizer used for encoding the data.
        instruction_prefix: The prefix marking the beginning of the user instruction.
        response_prefix: The prefix marking the beginning of the assistant response.
        debug: Whether to enable logging of tokenized examples for debugging.
        """
        self._default_collator = trl.DataCollatorForCompletionOnlyLM(
            tokenizer=tokenizer,
            instruction_template=instruction_prefix,
            response_template=response_prefix,
        )

        if not hasattr(tokenizer, "pad_token_id") or tokenizer.pad_token_id is None:
            raise RuntimeError("Tokenizer doesn't define `pad_token_id`.")

        self._debug = debug
        self._tokenizer = tokenizer

    def _collate(self, inputs: list[Any]) -> dict[str, Any]:
        result = self._default_collator(inputs)
        return result

    def __call__(self, batch) -> dict[str, Any]:
        """Pads to the longest length present in the batch.

        Args:
            batch: List of batch items.

        Returns:
            Dict[str, torch.Tensor]: Processed batch.
        """
        for item in batch:
            if _INPUT_IDS_KEY not in item:
                raise ValueError(
                    f"Item doesn't contain '{_INPUT_IDS_KEY}' key. "
                    f"Available keys: {item.keys()}"
                )

        # Collate batch prompts.
        collated_text_inputs = self._collate(batch)

        if self._debug:
            raw_example = batch[0]
            formatted_example = self._tokenizer.apply_chat_template(raw_example, tokenize=False)
            tokenized_example = self._tokenizer.apply_chat_template(raw_example)
            decoded_tokens = [self._tokenizer.decode(t) for t in tokenized_example]
            tokenized_example = list(zip(tokenized_example, decoded_tokens))
            model_input = collated_text_inputs
            log_tokenized_example(raw_example, formatted_example, tokenized_example, model_input)

        return collated_text_inputs
