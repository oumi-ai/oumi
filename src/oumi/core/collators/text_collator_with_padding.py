from typing import NamedTuple, Optional

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
        max_length: Optional[int],
        label_ignore_index: Optional[int] = -100,
    ):
        """Custom collator for multi-modal vision-language training."""
        self._default_collator = transformers.DataCollatorWithPadding(
            tokenizer=tokenizer,
            max_length=max_length,
            padding=True,
        )

        if not hasattr(tokenizer, "pad_token_id") or tokenizer.pad_token_id is None:
            raise RuntimeError("Tokenizer doesn't define `pad_token_id`.")

        self._special_tokens: _SpecialTokens = _SpecialTokens(
            pad_token_id=int(tokenizer.pad_token_id),
            label_ignore_index=label_ignore_index,
        )

    def __call__(self, batch):
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
            # Ignore `pad_token_id`-s in the loss computation.
            if self._special_tokens.label_ignore_index is not None:
                labels[labels == self._special_tokens.pad_token_id] = int(
                    self._special_tokens.label_ignore_index
                )
            combined_batch[_LABELS_KEY] = labels

        return combined_batch
