from typing import Optional

from transformers import SpecialTokensMixin

from oumi.core.tokenizers import BaseTokenizer
from oumi.utils.logging import logger

special_tokens = {
    "llama": SpecialTokensMixin(pad_token="<|finetune_right_pad_id|>"),
    "gpt2": SpecialTokensMixin(
        pad_token="<|pad|>"  # GPT2 has no padding token; this is defined by Oumi.
    ),
}


def get_default_special_tokens(
    tokenizer: Optional[BaseTokenizer],
) -> SpecialTokensMixin:
    """Returns the default special tokens for the tokenizer that was provided."""
    if tokenizer and tokenizer.name_or_path:
        if "llama" in tokenizer.name_or_path:
            return special_tokens["llama"]
        elif "gpt2" in tokenizer.name_or_path:
            return special_tokens["gpt2"]
        else:
            logger.warning(
                f"Special tokens lookup for tokenizer {tokenizer.name_or_path} failed."
            )
    return SpecialTokensMixin()
