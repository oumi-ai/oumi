from typing import Any, Dict, NamedTuple, Optional

import torch
import transformers

from oumi.core.tokenizers.base_tokenizer import BaseTokenizer

_INPUT_IDS_KEY = "input_ids"
_ATTENTION_MASK_KEY = "attention_mask"
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
            max_length=max_length,
            padding=("max_length" if self._max_length is not None else "longest"),
            return_tensors="pt",
        )

        if not hasattr(tokenizer, "pad_token_id") or tokenizer.pad_token_id is None:
            raise RuntimeError("Tokenizer doesn't define `pad_token_id`.")

        self._special_tokens: _SpecialTokens = _SpecialTokens(
            pad_token_id=int(tokenizer.pad_token_id),
            label_ignore_index=label_ignore_index,
        )

    def __call__(self, batch) -> Dict[str, Any]:
        """Pads to the longest length present in the batch.

        Args:
            batch: List of batch items.

        Returns:
            Dict[str, torch.Tensor]: Processed batch.
        """
        text_inputs = []
        labels = []
        labels_present = _LABELS_KEY in batch[0]
        for item in batch:
            if _INPUT_IDS_KEY not in item:
                raise ValueError(
                    f"Item doesn't contain '{_INPUT_IDS_KEY}' key. "
                    f"Available keys: {item.keys()}"
                )
            if self._max_length is not None and self._truncation:
                # Truncate to max length.
                text_inputs.append(item[_INPUT_IDS_KEY][0 : self._max_length])
                if labels_present:
                    labels.append(item[_LABELS_KEY][0 : self._max_length])
            else:
                text_inputs.append(item[_INPUT_IDS_KEY])
                if labels_present:
                    labels.append(item[_LABELS_KEY])

        # Collate batch prompts.
        collated_text_inputs = self._default_collator({_INPUT_IDS_KEY: text_inputs})  # type: ignore

        # Combine all inputs.
        combined_batch = {
            _INPUT_IDS_KEY: collated_text_inputs[_INPUT_IDS_KEY],
            _ATTENTION_MASK_KEY: collated_text_inputs.get(_ATTENTION_MASK_KEY),
        }

        # Add labels if present.
        if labels_present:
            collated_labels = self._default_collator({_INPUT_IDS_KEY: labels})  # type: ignore
            labels = collated_labels[_INPUT_IDS_KEY]
            assert isinstance(labels, torch.Tensor)
            # Ignore `pad_token_id`-s in the loss computation.
            if self._special_tokens.label_ignore_index is not None:
                labels[labels == self._special_tokens.pad_token_id] = int(
                    self._special_tokens.label_ignore_index
                )
            combined_batch[_LABELS_KEY] = labels

        return combined_batch
