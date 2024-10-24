import collections
from typing import Any, Dict, List, NamedTuple, Optional, Set

import torch
import transformers

from oumi.core.tokenizers.base_tokenizer import BaseTokenizer
from oumi.utils.logging import logger
from oumi.utils.torch_utils import pad_sequences

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
        """
        self._max_length: Optional[int] = (
            int(max_length) if max_length is not None and max_length > 0 else None
        )
        self._truncation: bool = bool(truncation)

        self._default_collator = transformers.DataCollatorWithPadding(
            tokenizer=tokenizer,
            max_length=self._max_length,
            padding="longest",
            return_tensors="pt",
        )

        if not hasattr(tokenizer, "padding_side") or not tokenizer.padding_side:
            raise RuntimeError("Tokenizer doesn't define `padding_side`.")
        self._padding_side = str(tokenizer.padding_side)

        if not hasattr(tokenizer, "pad_token_id") or tokenizer.pad_token_id is None:
            raise RuntimeError("Tokenizer doesn't define `pad_token_id`.")

        self._special_tokens: _SpecialTokens = _SpecialTokens(
            pad_token_id=int(tokenizer.pad_token_id),
            label_ignore_index=label_ignore_index,
        )

        self._max_input_ids_length: int = 0
        self._max_labels_length: int = 0
        self._max_previously_logged_input_ids_length: int = 0
        self._max_previously_logged_labels_length: int = 0

    def _collate(
        self,
        inputs_dict: Dict[str, List[Any]],
        batch_max_length: int,
        nested_input_names: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        nested_inputs_dict: Optional[Dict[str, List[Any]]] = None
        if nested_input_names is not None and len(nested_input_names) > 0:
            basic_inputs_dict: Dict[str, List[Any]] = {}
            nested_inputs_dict = {}
            for key, values_list in inputs_dict.items():
                if key in nested_input_names:
                    nested_inputs_dict[key] = list(values_list)
                else:
                    basic_inputs_dict[key] = list(values_list)
        else:
            basic_inputs_dict = inputs_dict

        try:
            result = self._default_collator(basic_inputs_dict)  # type: ignore
        except ValueError:
            logger.error(
                "Failed to collate! "
                f"Batch maximum length: {batch_max_length}. "
                f"Maximum allowed length: {self._max_length}. "
                f"Truncation: {self._truncation}."
            )
            raise

        if nested_inputs_dict is not None:
            if self._padding_side != "right":
                raise RuntimeError(
                    f"Can't collate {key} if padding_side={self._padding_side}"
                )
            for key, values_list in nested_inputs_dict.items():
                result[key] = pad_sequences(
                    values_list, padding_side=self._padding_side
                )

        return result

    def __call__(self, batch) -> Dict[str, Any]:
        """Pads to the longest length present in the batch.

        Args:
            batch: List of batch items.

        Returns:
            Dict[str, torch.Tensor]: Processed batch.
        """
        collation_inputs: Dict[str, List[Any]] = collections.defaultdict(list)
        labels = []
        labels_on = _LABELS_KEY in batch[0]
        cross_attention_mask_on = _CROSS_ATTENTION_MASK_KEY in batch[0]

        # Maximum sequence lengths in this batch.
        batch_max_input_ids_length: int = 0
        batch_max_labels_length: int = 0

        for item in batch:
            if _INPUT_IDS_KEY not in item:
                raise ValueError(
                    f"Item doesn't contain '{_INPUT_IDS_KEY}' key. "
                    f"Available keys: {item.keys()}"
                )
            batch_max_input_ids_length = max(
                batch_max_input_ids_length, len(item[_INPUT_IDS_KEY])
            )
            if labels_on:
                batch_max_labels_length = max(
                    batch_max_labels_length, len(item[_LABELS_KEY])
                )

            if self._max_length is not None and self._truncation:
                # Truncate to max length.
                collation_inputs[_INPUT_IDS_KEY].append(
                    item[_INPUT_IDS_KEY][0 : self._max_length]
                )
                if cross_attention_mask_on:
                    collation_inputs[_CROSS_ATTENTION_MASK_KEY].append(
                        item[_CROSS_ATTENTION_MASK_KEY][0 : self._max_length]
                    )
                if labels_on:
                    labels.append(item[_LABELS_KEY][0 : self._max_length])
            else:
                collation_inputs[_INPUT_IDS_KEY].append(item[_INPUT_IDS_KEY])
                if cross_attention_mask_on:
                    collation_inputs[_CROSS_ATTENTION_MASK_KEY].append(
                        item[_CROSS_ATTENTION_MASK_KEY]
                    )
                if labels_on:
                    labels.append(item[_LABELS_KEY])

        # Update global (dataset) maximum lengths, and log a warning
        # about truncation if needed.
        self._update_max_lengths_and_log(
            max_input_ids_length=batch_max_input_ids_length,
            max_labels_length=batch_max_labels_length,
        )

        # Collate batch prompts.
        collated_text_inputs = self._collate(
            collation_inputs,
            batch_max_length=batch_max_input_ids_length,
            nested_input_names=(
                {_CROSS_ATTENTION_MASK_KEY} if cross_attention_mask_on else None
            ),
        )

        # Combine all inputs.
        combined_batch = {
            _INPUT_IDS_KEY: collated_text_inputs[_INPUT_IDS_KEY],
            _ATTENTION_MASK_KEY: collated_text_inputs.get(_ATTENTION_MASK_KEY),
            _CROSS_ATTENTION_MASK_KEY: collated_text_inputs.get(
                _CROSS_ATTENTION_MASK_KEY
            ),
        }

        # Add labels if present.
        if labels_on:
            collated_labels = self._collate(
                {_INPUT_IDS_KEY: labels}, batch_max_length=batch_max_labels_length
            )
            labels = collated_labels[_INPUT_IDS_KEY]
            assert isinstance(labels, torch.Tensor)
            # Ignore `pad_token_id`-s in the loss computation.
            if self._special_tokens.label_ignore_index is not None:
                labels[labels == self._special_tokens.pad_token_id] = int(
                    self._special_tokens.label_ignore_index
                )
            combined_batch[_LABELS_KEY] = labels

        return combined_batch

    def _update_max_lengths_and_log(
        self, *, max_input_ids_length: int, max_labels_length: int
    ):
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

        if max_labels_length > self._max_labels_length:
            if (
                self._max_length is not None
                and self._max_labels_length > self._max_length
            ):
                if (
                    max_labels_length - self._max_previously_logged_labels_length
                ) >= _LOG_REL_INCREMENT * self._max_previously_logged_labels_length:
                    log_max_lengths = True
                    self._max_previously_logged_labels_length = max_labels_length
            self._max_labels_length = max_labels_length

        if log_max_lengths:
            logger.warning(
                "Input sequence exceeded max length"
                + (" and truncated! " if self._truncation else ". ")
                + (
                    f"Max allowed length: {self._max_length}. "
                    f"'input_ids' length: {self._max_input_ids_length}. "
                    f"'labels' length: {self._max_labels_length}."
                )
            )
