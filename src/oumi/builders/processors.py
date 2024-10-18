from typing import Callable

import transformers


def build_processor(
    model_name_or_path: str, *, trust_remote_code: bool = False
) -> Callable:
    """Builds a processor."""
    processor = transformers.AutoProcessor.from_pretrained(
        model_name_or_path, trust_remote_code=trust_remote_code
    )
    if not callable(processor):
        raise RuntimeError(
            f"Processor is not callable! Model name: '{model_name_or_path}'"
        )
    return processor
