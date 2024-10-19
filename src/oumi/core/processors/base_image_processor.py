import abc
from typing import List, Optional

import PIL.Image
import transformers


class BaseImageProcessor(abc.ABC):
    """Base class for oumi image processors."""

    @abc.abstractmethod
    def __call__(
        self,
        *,
        images: List[PIL.Image.Image],
        return_tensors: Optional[str] = "pt",
    ) -> transformers.BatchFeature:
        """Extracts image features."""
        raise NotImplementedError
