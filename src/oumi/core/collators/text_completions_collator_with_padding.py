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

import torch

from oumi.core.collators.trl_data_collator_for_completion_only_lm import (
    DataCollatorForCompletionOnlyLM,
)
from oumi.core.tokenizers.base_tokenizer import BaseTokenizer
from oumi.utils.debug_utils import log_example_for_debugging

_INPUT_IDS_KEY = "input_ids"


class TextCompletionsCollatorWithPadding:
    def __init__(
        self,
        tokenizer: BaseTokenizer,
        response_template: str,
        train_target: str,
        instruction_template: str | None = None,
        debug: bool = False,
        end_of_turn_template: str | None = None,
        ignore_index: int = -100,
        pad_to_multiple_of: int | None = None,
    ):
        """Custom collator for text LLM training.

        Args:
        tokenizer: The tokenizer used for encoding the data.
        response_template: String marking assistant response start.
        instruction_template: String marking user instruction start.
        debug: If True, enables debug mode for logging.
        train_target: Training target — ``"all_assistant_turns"``
            or ``"final_assistant_turn"``.
        end_of_turn_template: String marking the end of a turn.
            Required for ``all_assistant_turns``.
        ignore_index: Value used for masked labels. Must match the ignore_index
            of the loss function (default: -100).
        pad_to_multiple_of: If set, pad each batch up to a multiple of this
            value instead of exactly the longest sequence in the batch. Some
            compiled attention kernels (e.g. ``flex_attention``, block size
            128) cannot compile sequences shorter than one block; padding to
            the block size keeps short samples trainable. The extra positions
            carry ``labels=ignore_index`` and, under causal attention, are
            never attended by real tokens, so training is numerically
            unchanged.
        """
        self._default_collator = DataCollatorForCompletionOnlyLM(
            tokenizer=tokenizer,
            instruction_template=instruction_template,
            response_template=response_template,
            train_target=train_target,
            end_of_turn_template=end_of_turn_template,
            ignore_index=ignore_index,
        )

        if not hasattr(tokenizer, "pad_token_id") or tokenizer.pad_token_id is None:
            raise RuntimeError("Tokenizer doesn't define `pad_token_id`.")
        elif not isinstance(tokenizer.pad_token_id, int):
            raise RuntimeError(
                "Tokenizer's `pad_token_id` is not an integer. "
                f"{tokenizer.pad_token_id}. Type: {type(tokenizer.pad_token_id)}"
            )

        self._pad_to_multiple_of = pad_to_multiple_of
        self._pad_token_id = tokenizer.pad_token_id
        self._ignore_index = ignore_index
        self._padding_side = str(getattr(tokenizer, "padding_side", "right"))
        self._debug = debug
        self._has_logged_example = False

    def _collate(self, inputs: list[Any]) -> dict[str, Any]:
        result = self._default_collator(inputs)
        if self._pad_to_multiple_of:
            result = self._pad_batch_to_multiple(result)
        return result

    def _pad_batch_to_multiple(self, result: dict[str, Any]) -> dict[str, Any]:
        multiple = self._pad_to_multiple_of
        assert multiple is not None
        seq_len = result[_INPUT_IDS_KEY].shape[1]
        target = ((seq_len + multiple - 1) // multiple) * multiple
        extra = target - seq_len
        if extra == 0:
            return result

        def _extend(tensor: torch.Tensor, value: int) -> torch.Tensor:
            tail = tensor.new_full((tensor.shape[0], extra), value)
            return torch.cat([tensor, tail], dim=1)

        result[_INPUT_IDS_KEY] = _extend(result[_INPUT_IDS_KEY], self._pad_token_id)
        if "labels" in result:
            result["labels"] = _extend(result["labels"], self._ignore_index)
        if "attention_mask" not in result:
            return result

        # Any attention_mask — all-ones, or the [1..|0..|1..] of a mixed-length
        # batch — forces transformers to build per-batch mask closures that
        # break torch.compile caching and knock flex_attention off its fast
        # path. With right padding we can drop it entirely for sequential
        # position_ids: real tokens form a prefix, so they never attend padding
        # and their positions equal ``arange`` (identical numerics), padding
        # labels are ignore_index (no loss/grad), and TRL accepts position_ids
        # in lieu of a mask. Left padding needs the mask, so keep it there.
        if self._padding_side == "right":
            del result["attention_mask"]
            batch_size = result[_INPUT_IDS_KEY].shape[0]
            result["position_ids"] = (
                torch.arange(target, dtype=torch.long)
                .unsqueeze(0)
                .repeat(batch_size, 1)
            )
        else:
            result["attention_mask"] = _extend(result["attention_mask"], 1)
        return result

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
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
        # Decode() returns str | list[str]. For single sequences
        # (non-batched input), it always returns str. Assert this for type narrowing
        # to avoid type errors.
        assert isinstance(raw_text, str), "Expected str from decode for single sequence"
        assert isinstance(formatted_example, str)

        tokenized_example: list[tuple[int, str]] = []
        for token_id in token_ids:
            decoded = self._default_collator.tokenizer.decode([token_id])
            assert isinstance(decoded, str)
            tokenized_example.append((token_id, decoded))
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
