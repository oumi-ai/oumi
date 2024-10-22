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
        """Extracts image features.

        Args:
            images: A list of input images.
            return_tensors: The format of returned tensors.

        Returns:
            transformers.BatchFeature: The model-specific input features.
        """
        raise NotImplementedError
