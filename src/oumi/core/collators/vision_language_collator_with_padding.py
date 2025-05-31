# Copyright 2025 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Vision-Language collator for batching multimodal inputs.

This module provides a collator that handles both text and image data for
vision-language models. It extends the text collator functionality to properly
batch and pad both textual features (input_ids, attention_mask, labels) and
visual features (pixel_values, image masks, etc.).

Key Features:
    - Handles variable-sized images within batches
    - Supports models with single or multiple image inputs
    - Automatically pads sequences to consistent lengths
    - Preserves model-specific image features (masks, positions, etc.)
    - Integrates with the text collator for unified handling

Example:
    >>> from oumi.builders import build_tokenizer
    >>> from oumi.core.configs import ModelParams
    >>> tokenizer = build_tokenizer(ModelParams(model_name="llava-hf/llava-1.5-7b-hf"))
    >>> collator = VisionLanguageCollatorWithPadding(
    ...     tokenizer=tokenizer,
    ...     max_length=512,
    ...     truncation=True,
    ...     allow_multi_image_inputs=False
    ... )
    >>> batch = collator([
    ...     {"input_ids": [1, 2, 3], "pixel_values": image_tensor1},
    ...     {"input_ids": [4, 5], "pixel_values": image_tensor2}
    ... ])
"""

import collections
from typing import Any, Optional

import torch

from oumi.core.collators.text_collator_with_padding import TextCollatorWithPadding
from oumi.core.tokenizers.base_tokenizer import BaseTokenizer
from oumi.utils.torch_utils import pad_to_max_dim_and_stack


class VisionLanguageCollatorWithPadding:
    """Collator for vision-language models that handles both text and image batching.

    This collator extends TextCollatorWithPadding to handle multimodal inputs where
    each example contains both text tokens and image data. It properly batches and
    pads both modalities while preserving model-specific features.

    The collator expects each batch item to be a dictionary containing:
        - Text features: "input_ids", "attention_mask", "labels" (handled by text collator)
        - Image features: The main image feature (e.g., "pixel_values", "images")
        - Additional features: Model-specific features like "image_masks", "cross_attention_mask"

    Features are automatically collated based on their properties:
        - Text features are padded using the tokenizer's padding configuration
        - Image features are stacked or padded to handle variable sizes
        - Additional features are validated and stacked appropriately

    Note:
        This collator is typically used with models like LLAVA, BLIP, Qwen2-VL that
        process pre-extracted image features. For models requiring raw image processing,
        use VisionLanguageSftCollator instead.
    """

    def __init__(
        self,
        tokenizer: BaseTokenizer,
        *,
        max_length: Optional[int],
        truncation: bool = False,
        label_ignore_index: Optional[int] = None,
        allow_multi_image_inputs: bool = True,
        main_image_feature: str = "images",
    ):
        """Initialize the vision-language collator.

        Args:
            tokenizer: The tokenizer used for encoding text data. Must have valid
                padding_side and pad_token_id attributes.

            max_length: Maximum sequence length for padding. If None, sequences are
                padded to the longest sequence in the batch. If specified, shorter
                sequences are padded and longer sequences may be truncated based on
                the truncation parameter.

            truncation: Whether to truncate sequences longer than max_length.
                If False, long sequences are kept as-is even if they exceed max_length.
                Only takes effect when max_length is specified.

            label_ignore_index: Special value to mark tokens that should be ignored
                in loss computation (typically -100 for PyTorch). If set, padding tokens
                in labels are replaced with this value. Common values:
                - None: Keep original padding token IDs
                - -100: PyTorch's default ignore index for CrossEntropyLoss

            allow_multi_image_inputs: Whether the model supports multiple images per
                example. When True, allows variable number of images and adjusts padding
                dimensions accordingly. Models like MLLaMA support this, while others
                like early LLAVA versions only support single images.

            main_image_feature: The dictionary key for the main image data in each
                batch item. Common values:
                - "pixel_values": For models using raw pixel data (LLAVA, BLIP)
                - "images": For models using preprocessed features (Molmo)
                - "image_features": For models with pre-extracted features

        Raises:
            RuntimeError: If tokenizer lacks required padding_side or pad_token_id.
        """
        self._allow_multi_image_inputs = allow_multi_image_inputs
        self._main_image_feature = main_image_feature
        self._text_collator: TextCollatorWithPadding = TextCollatorWithPadding(
            tokenizer=tokenizer,
            max_length=max_length,
            truncation=truncation,
            label_ignore_index=label_ignore_index,
            max_variable_sized_dims=(
                # if multi-image inputs are possible, then
                # allow 2 variable-sized dimensions: `seq_len`, `num_images`.
                2 if allow_multi_image_inputs else 1
            ),
        )

    def __call__(self, batch) -> dict[str, Any]:
        """Process a batch of vision-language examples.

        This method orchestrates the collation of multimodal data by:
        1. Using the text collator to handle text features (input_ids, attention_mask, labels)
        2. Collecting and stacking image features from all examples
        3. Validating and stacking any additional model-specific features

        Args:
            batch: List of dictionaries, where each dictionary represents one example
                and must contain:
                - Text features handled by the text collator
                - The main image feature (as specified by main_image_feature)
                - Optional additional features (automatically detected and collated)

                Example format:
                [
                    {
                        "input_ids": [1, 2, 3, 4],
                        "attention_mask": [1, 1, 1, 1],
                        "labels": [1, 2, 3, 4],
                        "pixel_values": torch.tensor(...),  # shape: (C, H, W)
                        "image_masks": torch.tensor(...),    # optional
                    },
                    ...
                ]

        Returns:
            Dictionary containing all collated features with consistent tensor shapes:
                - "input_ids": Padded token IDs, shape (batch_size, max_seq_len)
                - "attention_mask": Attention masks, shape (batch_size, max_seq_len)
                - "labels": Padded labels with ignore indices, shape (batch_size, max_seq_len)
                - main_image_feature: Stacked images, shape depends on model and settings
                - Additional features: Any other features present in all examples

        Raises:
            ValueError: If any example is missing the main image feature, or if
                additional features are not present in all examples.
        """
        # Collate batch prompts
        collated_batch = self._text_collator(batch)  # type: ignore
        known_input_names: set[str] = set(collated_batch.keys()).union(
            {self._main_image_feature}
        )
        other_input_names: set[str] = set()

        images = []
        for item in batch:
            # TODO Consider relaxing this constraint: a vision/language model
            # can handle text-only inputs e.g., a follow-up to an answer,
            # or image-only inputs e.g., captioning.
            if self._main_image_feature not in item:
                raise ValueError(
                    f"Item doesn't contain '{self._main_image_feature}' key. "
                    f"Available keys: {item.keys()}"
                )
            images.append(item[self._main_image_feature])

            for key in item:
                if (
                    key
                    and (key not in known_input_names)
                    and (key not in other_input_names)
                ):
                    other_input_names.add(key)

        # Collate images.
        image_input_features = self.collate_images(images)

        # Add images to other inputs.
        collated_batch[self._main_image_feature] = image_input_features

        # For other inputs, let's verify they present in all examples and stack them.
        if len(other_input_names) > 0:
            other_inputs: dict[str, list[Any]] = collections.defaultdict(list)
            for item in batch:
                for input_name in other_input_names:
                    if input_name not in item:
                        raise ValueError(
                            f"Item doesn't contain '{input_name}' key. "
                            f"Available keys: {item.keys()}"
                        )
                    other_inputs[input_name].append(item[input_name])

            for input_name, values_list in other_inputs.items():
                collated_value = pad_to_max_dim_and_stack(
                    values_list,
                    max_variable_sized_dims=(
                        # if multi-image inputs are possible, then
                        # allow 1 variable-sized dimension (`num_images`).
                        1 if self._allow_multi_image_inputs else 0
                    ),
                )
                collated_batch[input_name] = collated_value

        return collated_batch

    def collate_images(self, images) -> torch.Tensor:
        """Collate image tensors into a batch with appropriate padding.

        This method handles the stacking of image tensors, which may have variable
        sizes depending on the model configuration. It uses intelligent padding to
        create consistent tensor shapes suitable for batch processing.

        Args:
            images: List of image tensors to collate. Each tensor can be:
                - For single image models: Shape (C, H, W) or similar
                - For multi-image models: Shape (num_images, C, H, W) or similar
                - May include additional dimensions for patches, features, etc.

        Returns:
            Batched image tensor with consistent shape across the batch dimension.
            The exact output shape depends on:
                - Input tensor shapes
                - allow_multi_image_inputs setting
                - Model-specific requirements

            Common output shapes:
                - Single image: (batch_size, C, H, W)
                - Multiple images: (batch_size, max_num_images, C, H, W)
                - With patches: (batch_size, num_patches, feature_dim)

        Raises:
            ValueError: If the images list is empty.

        Note:
            The method uses pad_to_max_dim_and_stack which intelligently handles
            variable-sized dimensions based on max_variable_sized_dims setting.
        """
        if len(images) == 0:
            raise ValueError("No images found in the batch")

        return pad_to_max_dim_and_stack(
            images,
            max_variable_sized_dims=(
                # if multi-image inputs are possible, then
                # allow 1 variable-sized dimension (`num_images`).
                1 if self._allow_multi_image_inputs else 0
            ),
        )
