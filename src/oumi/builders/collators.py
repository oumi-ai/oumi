import numpy as np
import torch
from transformers import DataCollatorWithPadding

_PIXEL_VALUES_KEY = "pixel_values"
_INPUT_IDS_KEY = "input_ids"
_ATTENTION_MASK_KEY = "attention_mask"
_LABELS_KEY = "labels"


def build_data_collator(collator_name: str, **kwargs):
    """Builds a data collator based on the given collator name.

    Args:
        collator_name: The name of the collator to build. Supported values are:
            - "text_with_padding": Uses DataCollatorWithPadding for text data.
            - "vision_language": Uses VisionLanguageCollator for multi-modal data.
        **kwargs: Additional keyword arguments to pass to the collator constructor.

    Returns:
        Callable: The data collator function or class.

    Raises:
        ValueError: If an unsupported collator name is provided.
    """
    if collator_name == "text_with_padding":
        return DataCollatorWithPadding(**kwargs)
    elif collator_name == "vision_language":
        return VisionLanguageCollator(**kwargs)

    return None


class VisionLanguageCollator:
    def __init__(self, processor, max_length: int = 1024):
        """Custom collator for multi-modal vision-language training."""
        self.processor = processor

        self._default_collator = DataCollatorWithPadding(
            tokenizer=self.processor.tokenizer,
            max_length=max_length,
            padding=True,
        )

    def __call__(self, batch):
        """Custom collator for multi-modal  vision-language training.

        Args:
            batch: List of batch items.

        Returns:
            Dict[str, torch.Tensor]: Processed batch.
        """
        images = []
        text_inputs = []
        for item in batch:
            # TODO Consider relaxing this constraint: a vision/language model
            # can handle text-only inputs e.g., a follow-up to an answer,
            # or image-only inputs e.g., captioning.
            for required_key in (_PIXEL_VALUES_KEY, _INPUT_IDS_KEY):
                if required_key not in item:
                    raise ValueError(
                        f"Item doesn't contain '{required_key}' key. "
                        f"Available keys: {item.keys()}"
                    )
            images.append(item[_PIXEL_VALUES_KEY])
            text_inputs.append(item[_INPUT_IDS_KEY])

        # collate batch images
        pixel_values = self.collate_images(images)

        # collate batch prompts
        text_inputs = self._default_collator({_INPUT_IDS_KEY: text_inputs})  # type: ignore

        # Combine all inputs
        combined_batch = {
            _PIXEL_VALUES_KEY: pixel_values,
            _INPUT_IDS_KEY: text_inputs[_INPUT_IDS_KEY],
            _ATTENTION_MASK_KEY: text_inputs.get(_ATTENTION_MASK_KEY),
        }

        # Add labels if present
        if _LABELS_KEY in batch[0]:
            combined_batch[_LABELS_KEY] = text_inputs[_INPUT_IDS_KEY]

        return combined_batch

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
