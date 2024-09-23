import numpy as np
import torch
from transformers import DataCollatorWithPadding


def build_data_collator(collator_name: str, **kwargs):
    """Build a data collator based on the given collator name.

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


class VisionLanguageCollator:
    def __init__(self, processor, max_length: int = 1024):
        """Custom collator for multi-modal vision-language training."""
        self.processor = processor

        self.default_collator = DataCollatorWithPadding(
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
        images = [item["pixel_values"] for item in batch]
        text_inputs = [item["input_ids"] for item in batch]

        # collate batch images
        pixel_values = self.collate_images(images)

        # collate batch prompts
        text_inputs = self.default_collator({"input_ids": text_inputs})  # type: ignore

        # Combine all inputs
        combined_batch = {
            "pixel_values": pixel_values,
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs.get("attention_mask"),
        }

        # Add labels if present
        if "labels" in batch[0]:
            combined_batch["labels"] = text_inputs["input_ids"]

        return combined_batch

    def collate_images(self, images) -> torch.Tensor:
        """Collate images for multi-modal training.

        Args:
            images: List of images to collate.

        Returns:
            torch.Tensor: Batch of processed images.
        """
        if isinstance(images[0], torch.Tensor):
            return torch.stack(images)
        elif isinstance(images[0], np.ndarray):
            return torch.stack([torch.from_numpy(img) for img in images])
        elif isinstance(images[0], list):
            return torch.tensor(images)
        else:
            raise ValueError(f"Unsupported image type: {type(images[0])}")
