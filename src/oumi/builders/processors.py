import transformers

from oumi.core.processors.base_processor import (
    BaseProcessor,
    DefaultProcessor,
)


def build_processor(
    model_name: str, *, trust_remote_code: bool = False
) -> BaseProcessor:
    """Builds a processor."""
    if not model_name:
        raise ValueError("Empty model name.")

    worker_processor = transformers.AutoProcessor.from_pretrained(
        model_name, trust_remote_code=trust_remote_code
    )

    return DefaultProcessor(worker_processor)
