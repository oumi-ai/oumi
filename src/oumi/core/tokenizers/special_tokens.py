from typing import Optional

from transformers import SpecialTokensMixin

from oumi.core.tokenizers import BaseTokenizer
from oumi.utils.logging import logger

special_tokens = {
    # Llama-3.1 models already have `<|finetune_right_pad_id|>` token in their vocab.
    "meta-llama/Llama-3.1-8B": SpecialTokensMixin(
        pad_token="<|finetune_right_pad_id|>"
    ),
    "meta-llama/Llama-3.1-8B-Instruct": SpecialTokensMixin(
        pad_token="<|finetune_right_pad_id|>"
    ),
    "meta-llama/Llama-3.1-70B": SpecialTokensMixin(
        pad_token="<|finetune_right_pad_id|>"
    ),
    "meta-llama/Llama-3.1-70B-Instruct": SpecialTokensMixin(
        pad_token="<|finetune_right_pad_id|>"
    ),
    "meta-llama/Llama-3.1-405B": SpecialTokensMixin(
        pad_token="<|finetune_right_pad_id|>"
    ),
    "meta-llama/Llama-3.1-405B-Instruct": SpecialTokensMixin(
        pad_token="<|finetune_right_pad_id|>"
    ),
    "meta-llama/Llama-3.1-405B-FP8": SpecialTokensMixin(
        pad_token="<|finetune_right_pad_id|>"
    ),
    "meta-llama/Llama-3.1-405B-Instruct-FP8": SpecialTokensMixin(
        pad_token="<|finetune_right_pad_id|>"
    ),
    # Llama-3.2 models already have `<|finetune_right_pad_id|>` token in their vocab.
    "meta-llama/Llama-3.2-1B": SpecialTokensMixin(
        pad_token="<|finetune_right_pad_id|>"
    ),
    "meta-llama/Llama-3.2-1B-Instruct": SpecialTokensMixin(
        pad_token="<|finetune_right_pad_id|>"
    ),
    "meta-llama/Llama-3.2-3B": SpecialTokensMixin(
        pad_token="<|finetune_right_pad_id|>"
    ),
    "meta-llama/Llama-3.2-3B-Instruct": SpecialTokensMixin(
        pad_token="<|finetune_right_pad_id|>"
    ),
}


def get_default_special_tokens(
    tokenizer: Optional[BaseTokenizer],
) -> SpecialTokensMixin:
    """Returns the default special tokens for the tokenizer that was provided.

    Args:
        tokenizer: The tokenizer to get special tokens for.

    Returns:
        The special tokens mixin for the tokenizer.

    Description:
        This function looks up the special tokens for the provided tokenizer, for a list
        of known models. If the tokenizer is not recognized, it returns an empty special
        tokens mixin. This function is used as a fallback mechanism when a special token
        is required, but is not provided in the tokenizer's configuration. The primary
        use case for this is to retrieve the padding special token (`pad_token`), which
        is oftentimes not included in the tokenizer's configuration, even if it exists
        in the tokenizer's vocabulary.
    """
    if tokenizer and tokenizer.name_or_path:
        if tokenizer.name_or_path in special_tokens:
            return special_tokens[tokenizer.name_or_path]
        else:
            logger.warning(
                f"Special tokens lookup for tokenizer {tokenizer.name_or_path} failed."
            )
    return SpecialTokensMixin()
