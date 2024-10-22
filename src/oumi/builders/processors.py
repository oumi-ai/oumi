import transformers

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

    worker_processor = transformers.AutoProcessor.from_pretrained(
        processor_name, trust_remote_code=trust_remote_code
    )

    return DefaultProcessor(worker_processor, tokenizer)
