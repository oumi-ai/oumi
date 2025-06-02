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

import collections
from typing import Any, NamedTuple, Optional

from oumi.core.tokenizers.base_tokenizer import BaseTokenizer
from oumi.utils.debug_utils import log_example_for_debugging
from oumi.utils.logging import logger
from oumi.utils.torch_utils import (
    create_ones_like,
    pad_sequences,
    pad_to_max_dim_and_stack,
)

_INPUT_IDS_KEY = "input_ids"
_ATTENTION_MASK_KEY = "attention_mask"
_CROSS_ATTENTION_MASK_KEY = "cross_attention_mask"
_LABELS_KEY = "labels"


class _SpecialTokens(NamedTuple):
    """Special tokens used by VisionLanguageCollatorWithPadding."""

    pad_token_id: int
    """Token id of `PAD` token."""

    label_ignore_index: Optional[int]
    """If set, then `PAD` tokens will be replaced by this special value
    to exclude them from the loss computation.
    """


class TextCollatorWithPadding:
    def __init__(
        self,
        tokenizer: BaseTokenizer,
        *,
        max_length: Optional[int],
        truncation: bool = False,
        label_ignore_index: Optional[int] = None,
        max_variable_sized_dims: int = 1,
        debug: bool = False,
    ):
        """Custom collator for text LLM training.

        Args:
        tokenizer: The tokenizer used for encoding the data.
        max_length: Padding length.
        truncation: Whether to truncate long inputs to `max_length`.
            If False, the long inputs are preserved as is even if they exceed
            `max_length`. Only has effect if `max_length` is specified.
        label_ignore_index:  If set, then label values of tokens that shouldn't
            contribute to the loss computation will be replaced by this special value.
        max_variable_sized_dims: Maximum number of variable-sized dimensions.
            Normally, it's 1 (sequence length dimension), but can sometimes be higher
            e.g., 2 for "cross_attention_mask" for VLM-s with multi-image inputs.
            Negative value mean `Unlimited`.
        debug: Whether to log a debug example.
        """
        self._max_length: Optional[int] = (
            int(max_length) if max_length is not None and max_length > 0 else None
        )
        self._truncation: bool = bool(truncation)

        if not hasattr(tokenizer, "padding_side") or not tokenizer.padding_side:
            raise RuntimeError("Tokenizer doesn't define `padding_side`.")
        self._padding_side = str(tokenizer.padding_side)

        if not hasattr(tokenizer, "pad_token_id") or tokenizer.pad_token_id is None:
            raise RuntimeError("Tokenizer doesn't define `pad_token_id`.")
        elif not isinstance(tokenizer.pad_token_id, int):
            raise RuntimeError(
                "Tokenizer's `pad_token_id` is not an integer. "
                f"{tokenizer.pad_token_id}. Type: {type(tokenizer.pad_token_id)}"
            )

        self._special_tokens: _SpecialTokens = _SpecialTokens(
            pad_token_id=int(tokenizer.pad_token_id),
            label_ignore_index=label_ignore_index,
        )

        self._max_input_ids_length: int = 0
        self._max_previously_logged_input_ids_length: int = 0
        self._max_variable_sized_dims: int = max_variable_sized_dims
        self._debug: bool = debug
        # Track if we've already logged an example
        self._has_logged_example: bool = False
        self._tokenizer = tokenizer  # Store tokenizer for debugging

    def _collate_simple(
        self,
        inputs_dict: dict[str, list[Any]],
        *,
        batch_max_length: int,
        padding_value_overrides: dict[str, int],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, sequences_list in inputs_dict.items():
            try:
                padding_value = padding_value_overrides.get(key, 0)
                if self._max_variable_sized_dims == 1:
                    collated_tensor = pad_sequences(
                        sequences_list,
                        padding_side=self._padding_side,
                        padding_value=padding_value,
                    )
                else:
                    collated_tensor = pad_to_max_dim_and_stack(
                        sequences_list,
                        max_variable_sized_dims=self._max_variable_sized_dims,
                        padding_side=self._padding_side,
                        padding_value=padding_value,
                    )
                result[key] = collated_tensor
            except Exception:
                logger.error(
                    f"Failed to collate '{key}'!  "
                    f"Max variable size dims: {self._max_variable_sized_dims}, "
                    f"Batch maximum length: {batch_max_length}, "
                    f"Maximum allowed length: {self._max_length}, "
                    f"Truncation: {self._truncation}."
                )
                raise
        return result

    def __call__(self, batch) -> dict[str, Any]:
        """Pads to the longest length present in the batch.

        Args:
            batch: List of batch items.

        Returns:
            Dict[str, torch.Tensor]: Processed batch.
        """
        collation_inputs: dict[str, list[Any]] = collections.defaultdict(list)
        labels_on = _LABELS_KEY in batch[0]
        attention_mask_on = _ATTENTION_MASK_KEY in batch[0]
        cross_attention_mask_on = _CROSS_ATTENTION_MASK_KEY in batch[0]

        # Maximum sequence lengths in this batch.
        batch_max_input_ids_length: int = 0

        for item in batch:
            if _INPUT_IDS_KEY not in item:
                raise ValueError(
                    f"Item doesn't contain '{_INPUT_IDS_KEY}' key. "
                    f"Available keys: {item.keys()}"
                )
            batch_max_input_ids_length = max(
                batch_max_input_ids_length, len(item[_INPUT_IDS_KEY])
            )

            collation_inputs[_INPUT_IDS_KEY].append(item[_INPUT_IDS_KEY])
            collation_inputs[_ATTENTION_MASK_KEY].append(
                item[_ATTENTION_MASK_KEY]
                if attention_mask_on
                else create_ones_like(item[_INPUT_IDS_KEY])
            )

            if cross_attention_mask_on:
                collation_inputs[_CROSS_ATTENTION_MASK_KEY].append(
                    item[_CROSS_ATTENTION_MASK_KEY]
                )
            if labels_on:
                collation_inputs[_LABELS_KEY].append(item[_LABELS_KEY])

            if self._max_length is not None:
                if self._truncation:
                    for key in collation_inputs:
                        collation_inputs[key] = [
                            item[0 : self._max_length] for item in collation_inputs[key]
                        ]
                else:
                    for key in collation_inputs:
                        for item in collation_inputs[key]:
                            seq_len = len(item)
                            if seq_len > self._max_length:
                                raise ValueError(
                                    "Maximum sequence length exceeded. "
                                    "You should probably activate truncation. "
                                    f"'{key}' length: ({seq_len}). "
                                    f"Maximum model length: ({self._max_length})"
                                )

        # Update global (dataset) maximum lengths, and log a warning
        # about truncation if needed.
        self._update_max_lengths_and_log(
            max_input_ids_length=batch_max_input_ids_length
        )

        # Collate batch prompts.
        pad_token_id = self._special_tokens.pad_token_id
        collated_text_inputs = self._collate_simple(
            collation_inputs,
            batch_max_length=batch_max_input_ids_length,
            padding_value_overrides={
                _INPUT_IDS_KEY: pad_token_id,
                _LABELS_KEY: (
                    self._special_tokens.label_ignore_index
                    if self._special_tokens.label_ignore_index is not None
                    else pad_token_id
                ),
            },
        )

        # Combine all inputs.
        combined_batch = {
            _INPUT_IDS_KEY: collated_text_inputs[_INPUT_IDS_KEY],
            _ATTENTION_MASK_KEY: collated_text_inputs.get(_ATTENTION_MASK_KEY),
        }

        if cross_attention_mask_on:
            combined_batch[_CROSS_ATTENTION_MASK_KEY] = collated_text_inputs[
                _CROSS_ATTENTION_MASK_KEY
            ]

        # Add labels if present.
        if labels_on:
            combined_batch[_LABELS_KEY] = collated_text_inputs[_LABELS_KEY]

        # If debug is on and we haven't logged an example yet, log the first example
        if self._debug and not self._has_logged_example and len(batch) > 0:
            # Log an example of the data in the first step for debugging purposes.
            self._log_debug_example(batch, combined_batch)

        return combined_batch

    def _log_debug_example(
        self,
        batch: list[dict[str, Any]],
        combined_batch: dict[str, Any],
    ) -> None:
        """Logs a debug example if debug is enabled.

        Args:
            batch: The original batch of data.
            combined_batch: The collated batch after processing.
        """
        first_input_ids = combined_batch[_INPUT_IDS_KEY][0]
        formatted_example = self._tokenizer.decode(
            first_input_ids, skip_special_tokens=False
        )
        # Decode raw text without special tokens for raw example
        raw_text = self._tokenizer.decode(first_input_ids, skip_special_tokens=True)

        tokenized_example = []
        for tid in first_input_ids:
            if hasattr(tid, "item"):
                token_id = int(tid.item())
                decoded_token = self._tokenizer.decode([tid])
            else:
                token_id = int(tid)
                decoded_token = self._tokenizer.decode(tid)
            tokenized_example.append((token_id, decoded_token))

        model_input = {
            "input_ids": (
                first_input_ids.tolist()
                if hasattr(first_input_ids, "tolist")
                else first_input_ids
            ),
            "attention_mask": (
                combined_batch[_ATTENTION_MASK_KEY][0].tolist()
                if hasattr(combined_batch[_ATTENTION_MASK_KEY][0], "tolist")
                else combined_batch[_ATTENTION_MASK_KEY][0]
            ),
        }

        if _LABELS_KEY in combined_batch:
            lbl = combined_batch[_LABELS_KEY][0]
            model_input["labels"] = lbl.tolist() if hasattr(lbl, "tolist") else lbl

        # Mark that we've logged an example to avoid logging again
        self._has_logged_example = True
        log_example_for_debugging(
            raw_example=raw_text,
            formatted_example=str(formatted_example),
            tokenized_example=tokenized_example,
            model_input=model_input,
        )

    def _update_max_lengths_and_log(self, *, max_input_ids_length: int):
        """Updates max length counters.

        Also, logs a truncation warning if increment is large enough.
        """
        _LOG_REL_INCREMENT = 0.1  # log if max length is up 10%
        log_max_lengths: bool = False

        if max_input_ids_length > self._max_input_ids_length:
            if self._max_length is not None and max_input_ids_length > self._max_length:
                if (
                    max_input_ids_length - self._max_previously_logged_input_ids_length
                ) >= _LOG_REL_INCREMENT * self._max_previously_logged_input_ids_length:
                    log_max_lengths = True
                    self._max_previously_logged_input_ids_length = max_input_ids_length
            self._max_input_ids_length = max_input_ids_length

        if log_max_lengths:
            logger.warning(
                "Input sequence exceeded max length"
                + (" and truncated! " if self._truncation else ". ")
                + (
                    f"Max allowed length: {self._max_length}. "
                    f"'input_ids' length: {self._max_input_ids_length}."
                )
            )
