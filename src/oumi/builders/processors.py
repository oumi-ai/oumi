import functools
from typing import Optional

import transformers

from oumi.core.constants import LABEL_IGNORE_INDEX
from oumi.core.processors.base_processor import BaseProcessor
from oumi.core.processors.default_processor import DefaultProcessor
from oumi.core.tokenizers.base_tokenizer import BaseTokenizer


def build_processor(
    processor_name: str, tokenizer: BaseTokenizer, *, trust_remote_code: bool = False
) -> BaseProcessor:
    """Builds a processor.

    Args:
        processor_name: A name of the processor (usually, equals to a model name).
        tokenizer: A tokenizer to use with the processor.
        trust_remote_code: Whether to allow loading remote code for this processor
            Some processors come with downloadable executable Python files,
            which can be a potential security risk, unless it's from a trusted source.

    Returns:
        BaseProcessor: The newly created processor.
    """
    if not processor_name:
        raise ValueError("Empty model name.")

    label_ignore_index: Optional[int] = LABEL_IGNORE_INDEX
    create_processor_fn = functools.partial(
        transformers.AutoProcessor.from_pretrained,
        processor_name,
        trust_remote_code=trust_remote_code,
    )
    # TODO OPE-701 Replace the special cases with a more general mechanism.
    if processor_name == "llava-hf/llava-1.5-7b-hf":
        worker_processor = create_processor_fn(
            patch_size=14, vision_feature_select_strategy="default"
        )
    elif processor_name == "Salesforce/blip2-opt-2.7b":
        worker_processor = create_processor_fn(num_query_tokens=32)
    elif processor_name == "microsoft/Phi-3-vision-128k-instruct":
        label_ignore_index = None
        worker_processor = create_processor_fn()
    else:
        worker_processor = create_processor_fn()

    return DefaultProcessor(
        worker_processor, tokenizer, label_ignore_index=label_ignore_index
    )
