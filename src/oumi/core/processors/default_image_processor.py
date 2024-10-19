from typing import Any, Callable, List, Optional

import PIL.Image
import transformers
from typing_extensions import override

from oumi.core.processors.base_image_processor import BaseImageProcessor


class DefaultImageProcessor(BaseImageProcessor):
    """Default implementation of image processor that wraps a callable function."""

    def __init__(self, worker_processor: Any):
        """Initializes the processor."""
        if worker_processor is None:
            raise ValueError("Worker image processor must be provided!")
        elif not callable(worker_processor):
            raise ValueError("Worker image processor is not callable!")
        self._worker_processor: Callable = worker_processor

    @override
    def __call__(
        self,
        *,
        images: List[PIL.Image.Image],
        return_tensors: Optional[str] = "pt",
    ) -> transformers.BatchFeature:
        """Extracts image features."""
        result = self._worker_processor(images=images, return_tensors=return_tensors)
        if result is None:
            raise RuntimeError("Image processor returned `None`.")
        elif not isinstance(result, transformers.BatchFeature):
            raise RuntimeError(
                "Image processor returned an object that is not a BatchFeature. "
                f"Actual type: {type(result)}"
            )
        return result
