import transformers

from oumi.core.processors.base_processor import (
    BaseProcessor,
    DefaultProcessor,
)


def build_processor(
    model_name_or_path: str, *, trust_remote_code: bool = False
) -> BaseProcessor:
    """Builds a processor."""
    worker_processor = transformers.AutoProcessor.from_pretrained(
        model_name_or_path, trust_remote_code=trust_remote_code
    )

    return DefaultProcessor(worker_processor)
