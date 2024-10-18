import transformers

from oumi.core.processors.base_processor import (
    BaseProcessor,
    DefaultProcessor,
)
from oumi.core.tokenizers.base_tokenizer import BaseTokenizer


def build_processor(
    model_name: str, tokenizer: BaseTokenizer, *, trust_remote_code: bool = False
) -> BaseProcessor:
    """Builds a processor."""
    if not model_name:
        raise ValueError("Empty model name.")

    worker_processor = transformers.AutoProcessor.from_pretrained(
        model_name, trust_remote_code=trust_remote_code
    )

    return DefaultProcessor(worker_processor, tokenizer)
