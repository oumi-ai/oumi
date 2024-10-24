from typing import Any, Optional

import numpy as np
import torch

from oumi.core.collators.text_collator_with_padding import TextCollatorWithPadding
from oumi.core.tokenizers.base_tokenizer import BaseTokenizer

_PIXEL_VALUES_KEY = "pixel_values"


class VisionLanguageCollatorWithPadding:
    def __init__(
        self,
        tokenizer: BaseTokenizer,
        *,
        max_length: Optional[int],
        truncation: bool = False,
        label_ignore_index: Optional[int] = None,
    ):
        """Custom collator for multi-modal vision-language training.

        Args:
        tokenizer: The tokenizer used for encoding the data.
        max_length: Padding length.
        truncation: Whether to truncate long inputs to `max_length`.
            If False, the long inputs are preserved as is even if they exceed
            `max_length`. Only has effect if `max_length` is specified.
        label_ignore_index:  If set, then label values of tokens that shouldn't
            contribute to the loss computation will be replaced by this special value.
        """
        self._text_collator = TextCollatorWithPadding(
            tokenizer=tokenizer,
            max_length=max_length,
            truncation=truncation,
            label_ignore_index=label_ignore_index,
        )

    def __call__(self, batch) -> dict[str, Any]:
        """Custom collator for multi-modal  vision-language training.

        Args:
            batch: List of batch items.

        Returns:
            Dict[str, torch.Tensor]: Processed batch.
        """
        # Collate batch prompts
        collated_batch = self._text_collator(batch)  # type: ignore

        images = []
        for item in batch:
            # TODO Consider relaxing this constraint: a vision/language model
            # can handle text-only inputs e.g., a follow-up to an answer,
            # or image-only inputs e.g., captioning.
            if _PIXEL_VALUES_KEY not in item:
                raise ValueError(
                    f"Item doesn't contain '{_PIXEL_VALUES_KEY}' key. "
                    f"Available keys: {item.keys()}"
                )
            images.append(item[_PIXEL_VALUES_KEY])

        # Collate batch images.
        pixel_values = self.collate_images(images)

        # Add images to other inputs.
        collated_batch[_PIXEL_VALUES_KEY] = pixel_values

        return collated_batch

    def collate_images(self, images) -> torch.Tensor:
        """Collate images for multi-modal training.

        Args:
            images: List of images to collate.

        Returns:
            torch.Tensor: Batch of processed images.
        """
        if len(images) == 0:
            raise ValueError("No images found in the batch")

        if isinstance(images[0], torch.Tensor):
            return torch.stack(images)
        elif isinstance(images[0], np.ndarray):
            return torch.stack([torch.from_numpy(img) for img in images])
        elif isinstance(images[0], list):
            return torch.tensor(images)
        else:
            raise ValueError(f"Unsupported image type: {type(images[0])}")
