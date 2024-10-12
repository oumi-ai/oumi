from typing import Optional

import numpy as np
import torch

from oumi.core.collators.text_collator_with_padding import TextCollatorWithPadding
from oumi.core.tokenizers.base_tokenizer import BaseTokenizer

_PIXEL_VALUES_KEY = "pixel_values"


class VisionLanguageCollatorWithPadding:
    def __init__(
        self,
        tokenizer: BaseTokenizer,
        max_length: Optional[int],
        label_ignore_index: Optional[int] = -100,
    ):
        """Custom collator for multi-modal vision-language training."""
        self._text_collator = TextCollatorWithPadding(
            tokenizer=tokenizer,
            max_length=max_length,
            label_ignore_index=label_ignore_index,
        )

    def __call__(self, batch):
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
