from typing import Callable, Optional

from oumi.core.collators.text_collator_with_padding import TextCollatorWithPadding
from oumi.core.collators.vision_language_collator_with_padding import (
    VisionLanguageCollatorWithPadding,
)
from oumi.core.tokenizers.base_tokenizer import BaseTokenizer


def build_data_collator(
    collator_name: str,
    tokenizer: BaseTokenizer,
    *,
    max_length: Optional[int],
    label_ignore_index: Optional[int],
    **kwargs,
) -> Callable:
    """Builds a data collator based on the given collator name.

    Args:
        collator_name: The name of the collator to build. Supported values are:
            - "text_with_padding": Uses TextCollatorWithPadding for text data.
            - "vision_language_with_padding": Uses VisionLanguageCollatorWithPadding
                for multi-modal data.
        tokenizer: A tokenizer.
        max_length: An optional maximum sequence length.
        label_ignore_index: If set, then label values of tokens that shouldn't
            contribute to the loss computation will be replaced by this special value.
            For example, this can be `PAD`, or image tokens.
            PyTorch convention is to use -100 as the `ignore_index` label. Refer to
            the `ignore_index` parameter of `torch.nn.CrossEntropyLoss()`
            for more details.
        **kwargs: Additional keyword arguments to pass to the collator constructor.

    Returns:
        Callable: The data collator function or class.

    Raises:
        ValueError: If an unsupported collator name is provided.
    """
    if not collator_name:
        raise ValueError("Empty data collator name.")

    if collator_name == "text_with_padding":
        return TextCollatorWithPadding(
            tokenizer=tokenizer,
            max_length=max_length,
            label_ignore_index=label_ignore_index,
            **kwargs,
        )
    elif collator_name == "vision_language_with_padding":
        return VisionLanguageCollatorWithPadding(
            tokenizer=tokenizer,
            max_length=max_length,
            label_ignore_index=label_ignore_index,
            **kwargs,
        )

    raise ValueError(f"Unknown data collator name: '{collator_name}'")
