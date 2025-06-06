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
from oumi.utils.debug_utils import log_example_for_debugging

_INPUT_IDS_KEY = "input_ids"


class TextCompletionsCollatorWithPadding:
    def __init__(
        self,
        tokenizer: BaseTokenizer,
        instruction_prefix: str,
        response_prefix: str,
        debug: bool = False,
    ):
        """Custom collator for text LLM training.

        Args:
        tokenizer: The tokenizer used for encoding the data.
        instruction_prefix: The prefix marking the beginning of the user instruction.
        response_prefix: The prefix marking the beginning of the assistant response.
        debug: If True, enables debug mode for logging.
        """
        self._default_collator = trl.DataCollatorForCompletionOnlyLM(
            tokenizer=tokenizer,
            instruction_template=instruction_prefix,
            response_template=response_prefix,
        )

        if not hasattr(tokenizer, "pad_token_id") or tokenizer.pad_token_id is None:
            raise RuntimeError("Tokenizer doesn't define `pad_token_id`.")

        self._debug = debug
        self._has_logged_example = False

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

        if self._debug and not self._has_logged_example:
            # Log an example of the data in the first step for debugging purposes.
            self._log_debug_example(batch, collated_text_inputs)
        return collated_text_inputs

    def _log_debug_example(
        self, batch: list[dict[str, Any]], collated_text_inputs: dict[str, Any]
    ) -> None:
        """Logs an example of the data in each step for debugging purposes.

        Args:
            batch: The batch of examples to log.
            collated_text_inputs: The collated inputs after processing.
        """
        raw_example = batch[0]
        token_ids = raw_example[_INPUT_IDS_KEY]
        # Raw text without special tokens
        raw_text = self._default_collator.tokenizer.decode(
            token_ids, skip_special_tokens=True
        )
        # Formatted example with special tokens
        formatted_example = self._default_collator.tokenizer.decode(
            token_ids, skip_special_tokens=False
        )
        tokenized_ids = raw_example[_INPUT_IDS_KEY]
        tokenized_example = [
            (token_id, self._default_collator.tokenizer.decode([token_id]))
            for token_id in tokenized_ids
        ]
        self._has_logged_example = True

        # Extract the first example from the batched tensors for cleaner debug output
        def _to_py(x):
            """Convert tensor-like objects to Python native types."""
            if hasattr(x, "tolist"):
                return x.tolist()
            elif hasattr(x, "item"):
                return x.item()
            else:
                return x

        # Process the collated inputs to get a clean representation for debugging
        model_input = {}
        for key, value in collated_text_inputs.items():
            # For batch tensors, extract just the first example
            if hasattr(value, "dim") and value.dim() > 1:
                model_input[key] = _to_py(value[0])
            # For single tensors or other objects
            else:
                model_input[key] = _to_py(value)

        # Log all components for debugging
        log_example_for_debugging(
            raw_text, formatted_example, tokenized_example, model_input
        )
