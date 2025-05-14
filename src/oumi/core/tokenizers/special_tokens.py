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

from typing import Optional

from transformers import SpecialTokensMixin

from oumi.core.tokenizers import BaseTokenizer
from oumi.utils.logging import logger

# Llama 3.1/3.2 models already have `<|finetune_right_pad_id|>` token in their vocab.
LLAMA_SPECIAL_TOKENS_MIXIN = SpecialTokensMixin(pad_token="<|finetune_right_pad_id|>")

special_tokens = {
    "meta-llama/Llama-3.1-8B": LLAMA_SPECIAL_TOKENS_MIXIN,
    "meta-llama/Llama-3.1-8B-Instruct": LLAMA_SPECIAL_TOKENS_MIXIN,
    "meta-llama/Llama-3.1-70B": LLAMA_SPECIAL_TOKENS_MIXIN,
    "meta-llama/Llama-3.1-70B-Instruct": LLAMA_SPECIAL_TOKENS_MIXIN,
    "meta-llama/Llama-3.1-405B": LLAMA_SPECIAL_TOKENS_MIXIN,
    "meta-llama/Llama-3.1-405B-Instruct": LLAMA_SPECIAL_TOKENS_MIXIN,
    "meta-llama/Llama-3.1-405B-FP8": LLAMA_SPECIAL_TOKENS_MIXIN,
    "meta-llama/Llama-3.1-405B-Instruct-FP8": LLAMA_SPECIAL_TOKENS_MIXIN,
    "meta-llama/Meta-Llama-3.1-8B": LLAMA_SPECIAL_TOKENS_MIXIN,
    "meta-llama/Meta-Llama-3.1-8B-Instruct": LLAMA_SPECIAL_TOKENS_MIXIN,
    "meta-llama/Meta-Llama-3.1-70B": LLAMA_SPECIAL_TOKENS_MIXIN,
    "meta-llama/Meta-Llama-3.1-70B-Instruct": LLAMA_SPECIAL_TOKENS_MIXIN,
    "meta-llama/Meta-Llama-3.1-405B": LLAMA_SPECIAL_TOKENS_MIXIN,
    "meta-llama/Meta-Llama-3.1-405B-Instruct": LLAMA_SPECIAL_TOKENS_MIXIN,
    "meta-llama/Meta-Llama-3.1-405B-FP8": LLAMA_SPECIAL_TOKENS_MIXIN,
    "meta-llama/Meta-Llama-3.1-405B-Instruct-FP8": LLAMA_SPECIAL_TOKENS_MIXIN,
    "meta-llama/Llama-3.2-1B": LLAMA_SPECIAL_TOKENS_MIXIN,
    "meta-llama/Llama-3.2-1B-Instruct": LLAMA_SPECIAL_TOKENS_MIXIN,
    "meta-llama/Llama-3.2-3B": LLAMA_SPECIAL_TOKENS_MIXIN,
    "meta-llama/Llama-3.2-3B-Instruct": LLAMA_SPECIAL_TOKENS_MIXIN,
}

# Lowercase all keys for case-insensitive lookup.
special_tokens = {k.lower(): v for k, v in special_tokens.items()}


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
        if tokenizer.name_or_path.lower() in special_tokens:
            return special_tokens[tokenizer.name_or_path.lower()]
        else:
            logger.warning(
                f"Special tokens lookup for tokenizer {tokenizer.name_or_path} failed."
            )
    return SpecialTokensMixin()
