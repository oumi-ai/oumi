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

import collections
from typing import Any, Optional

import torch

from oumi.core.collators.text_collator_with_padding import TextCollatorWithPadding
from oumi.core.tokenizers.base_tokenizer import BaseTokenizer
from oumi.utils.logging import logger
from oumi.utils.torch_utils import pad_to_max_dim_and_stack


class VisionLanguageCollatorWithPadding:
    def __init__(
        self,
        tokenizer: BaseTokenizer,
        *,
        max_length: Optional[int],
        truncation: bool = False,
        label_ignore_index: Optional[int] = None,
        allow_multi_image_inputs: bool = True,
        main_image_feature: str = "pixel_values",
        debug: bool = False,
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
        allow_multi_image_inputs: Whether to allow multi-image inputs.
        main_image_feature: The key to use for fetching the main image data
        (e.g., raw pixels, patches, etc.) from the input.
        debug: Whether to log a debug example.
        """
        self._allow_multi_image_inputs = allow_multi_image_inputs
        self._main_image_feature = main_image_feature
        self._debug = debug
        self._has_logged_example = False
        self._tokenizer = tokenizer
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
            debug=False,  # We'll handle debug logging at this level
        )

    def __call__(self, batch) -> dict[str, Any]:
        """Custom collator for multi-modal vision-language training.

        Args:
            batch: List of batch items.

        Returns:
            Dict[str, torch.Tensor]: Processed batch.
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

        # Log debug example if enabled
        if self._debug and not self._has_logged_example and len(batch) > 0:
            self._log_multimodal_debug_example(batch, collated_batch)

        return collated_batch

    def _log_multimodal_debug_example(
        self,
        batch: list[dict[str, Any]],
        collated_batch: dict[str, Any],
    ) -> None:
        """Logs a multimodal debug example.

        Args:
            batch: The original batch of data.
            collated_batch: The collated batch after processing.
        """
        self._has_logged_example = True

        # Log text information
        first_input_ids = collated_batch["input_ids"][0]
        formatted_text = self._tokenizer.decode(
            first_input_ids, skip_special_tokens=False
        )
        raw_text = self._tokenizer.decode(first_input_ids, skip_special_tokens=True)

        # Log tokenized text
        tokenized_example = []
        for tid in first_input_ids:
            if hasattr(tid, "item"):
                token_id = int(tid.item())
                decoded_token = self._tokenizer.decode([tid])
            else:
                token_id = int(tid)
                decoded_token = self._tokenizer.decode([tid])
            tokenized_example.append((token_id, decoded_token))

        # Log image information
        image_info = {}
        if self._main_image_feature in collated_batch:
            image_tensor = collated_batch[self._main_image_feature]
            image_info["image_feature_key"] = self._main_image_feature
            image_info["image_tensor_shape"] = (
                tuple(image_tensor.shape)
                if hasattr(image_tensor, "shape")
                else "unknown"
            )
            image_info["image_tensor_dtype"] = (
                str(image_tensor.dtype)
                if hasattr(image_tensor, "dtype")
                else "unknown"
            )
            image_info["batch_size"] = image_tensor.shape[0] if len(image_tensor.shape) > 0 else 0

            # For multi-image inputs, log additional info
            if self._allow_multi_image_inputs and len(image_tensor.shape) > 1:
                image_info["num_images_first_example"] = (
                    image_tensor.shape[1] if len(image_tensor.shape) > 1 else 1
                )

        # Build model input dict
        model_input = {
            "input_ids": (
                first_input_ids.tolist()
                if hasattr(first_input_ids, "tolist")
                else first_input_ids
            ),
            "attention_mask": (
                collated_batch["attention_mask"][0].tolist()
                if hasattr(collated_batch["attention_mask"][0], "tolist")
                else collated_batch["attention_mask"][0]
            ),
        }

        if "labels" in collated_batch:
            lbl = collated_batch["labels"][0]
            model_input["labels"] = lbl.tolist() if hasattr(lbl, "tolist") else lbl

        # Log all debug information
        logger.debug("=" * 80)
        logger.debug("MULTIMODAL DEBUG EXAMPLE")
        logger.debug("=" * 80)
        logger.debug("Raw text: %s", raw_text)
        logger.debug("Formatted text: %s", formatted_text)
        logger.debug("Tokenized example (first 10 tokens): %s", tokenized_example[:10])
        logger.debug("Image information: %s", image_info)
        logger.debug("Model input keys: %s", list(model_input.keys()))
        logger.debug("Model input (truncated): input_ids length=%d, attention_mask length=%d",
                    len(model_input["input_ids"]),
                    len(model_input["attention_mask"]))

        # Log any additional features beyond standard keys
        standard_keys = {"input_ids", "attention_mask", "labels", self._main_image_feature}
        additional_features = set(collated_batch.keys()) - standard_keys
        if additional_features:
            logger.debug("Additional features in batch: %s", list(additional_features))
        logger.debug("=" * 80)

    def collate_images(self, images) -> torch.Tensor:
        """Collate images for multi-modal training.

        Args:
            images: List of images to collate.

        Returns:
            torch.Tensor: Batch of processed images.
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
