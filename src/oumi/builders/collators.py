from typing import Callable, Optional

import transformers

from oumi.core.collators.vision_language_collator_with_padding import (
    VisionLanguageCollatorWithPadding,
)
from oumi.core.tokenizers.base_tokenizer import BaseTokenizer


def build_data_collator(
    collator_name: str,
    tokenizer: BaseTokenizer,
    max_length: Optional[int],
    **kwargs,
) -> Callable:
    """Builds a data collator based on the given collator name.

    Args:
        collator_name: The name of the collator to build. Supported values are:
            - "text_with_padding": Uses DataCollatorWithPadding for text data.
            - "vision_language": Uses VisionLanguageCollator for multi-modal data.
        tokenizer: A tokenizer.
        max_length: An optional maximum sequence length.
        **kwargs: Additional keyword arguments to pass to the collator constructor.

    Returns:
        Callable: The data collator function or class.

    Raises:
        ValueError: If an unsupported collator name is provided.
    """
    if not collator_name:
        raise ValueError("Empty data collator name.")

    if collator_name == "text_with_padding":
        return transformers.DataCollatorWithPadding(
            tokenizer=tokenizer, max_length=max_length, **kwargs
        )
    elif collator_name == "vision_language_with_padding":
        return VisionLanguageCollatorWithPadding(
            tokenizer=tokenizer, max_length=max_length, **kwargs
        )

    raise ValueError(f"Unknown data collator name: '{collator_name}'")
