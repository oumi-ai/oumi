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

import numpy as np
import torch

from oumi.core.collators.text_collator_with_padding import TextCollatorWithPadding
from oumi.core.tokenizers.base_tokenizer import BaseTokenizer
from oumi.utils.logging import logger
from oumi.utils.torch_utils import convert_to_list_of_tensors

_PIXEL_VALUES_KEY = "pixel_values"


def _pad_1d_and_stack(tensors_list: list[torch.Tensor]) -> torch.Tensor:
    num_tensors = len(tensors_list)
    if num_tensors <= 0:
        raise ValueError("No tensors")
    elif num_tensors == 1:
        return torch.stack(tensors_list)

    first_shape = tensors_list[0].shape
    num_dims = len(first_shape)

    variable_dim_idx: int = -1
    variable_dim_max_size: int = 0

    for tensor_idx in range(num_tensors - 1):
        curr_shape = tensors_list[tensor_idx + 1].shape
        if num_dims != len(curr_shape):
            raise ValueError(
                "Tensors have different number of dimensions: "
                f"{num_dims} vs {len(curr_shape)}! "
                f"Shapes: {first_shape}, {curr_shape}"
            )

        if curr_shape != first_shape:
            for idx in range(num_dims):
                if first_shape[idx] != curr_shape[idx]:
                    if variable_dim_idx < 0:
                        variable_dim_idx = idx
                        variable_dim_max_size = max(curr_shape[idx], first_shape[idx])
                    elif variable_dim_idx == idx:
                        variable_dim_max_size = max(
                            curr_shape[idx], variable_dim_max_size
                        )
                    else:
                        raise ValueError(
                            "Multiple variable dimensions detected: "
                            f"{variable_dim_idx} and {idx}! "
                            f"Shapes: {first_shape}, {curr_shape}"
                        )

    if variable_dim_idx >= 0:  # Found 1 variable dimension.
        for tensor_idx in range(num_tensors):
            curr_tensor = tensors_list[tensor_idx]
            curr_shape = curr_tensor.shape
            curr_dim_size = curr_shape[variable_dim_idx]
            padding_len = variable_dim_max_size - curr_dim_size
            if padding_len > 0:
                padding_shape = list(curr_shape)
                padding_shape[variable_dim_idx] = padding_len
                zero_pad_tensor = torch.zeros(
                    padding_shape, dtype=curr_tensor.dtype, device=curr_tensor.device
                )
                tensors_list[tensor_idx] = torch.cat(
                    (curr_tensor, zero_pad_tensor), dim=variable_dim_idx
                )
                new_shape = tensors_list[tensor_idx].shape
                logger.warning(
                    f"Padded dimension {variable_dim_idx} from {curr_dim_size} "
                    f"to {variable_dim_max_size} elements. "
                    f"Shapes: from {curr_shape} to {new_shape}"
                )

    return torch.stack(tensors_list)


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
        self._text_collator: TextCollatorWithPadding = TextCollatorWithPadding(
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
        known_input_names: set[str] = set(collated_batch.keys()).union(
            {_PIXEL_VALUES_KEY}
        )
        other_input_names: set[str] = set()

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

            for key in item:
                if (
                    key
                    and (key not in known_input_names)
                    and (key not in other_input_names)
                ):
                    other_input_names.add(key)

        logger.info("===========================")

        # Collate images.
        pixel_values = self.collate_images(images)

        logger.info(
            f"Collated '{_PIXEL_VALUES_KEY}': {pixel_values.shape} from: "
            + ", ".join(f"{t.shape}" for t in images)
        )
        # Add images to other inputs.
        collated_batch[_PIXEL_VALUES_KEY] = pixel_values

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
                tensors_list = convert_to_list_of_tensors(values_list)
                collated_value = _pad_1d_and_stack(tensors_list)
                collated_batch[input_name] = collated_value

                logger.info(
                    f"Collated '{input_name}': {collated_value.shape} from: "
                    + ", ".join(f"{t.shape}" for t in tensors_list)
                )

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

        if isinstance(images[0], np.ndarray):
            images = [torch.from_numpy(img) for img in images]

        if isinstance(images[0], torch.Tensor):
            return _pad_1d_and_stack(images)
        elif isinstance(images[0], list):
            return torch.tensor(images)
        else:
            raise ValueError(f"Unsupported image type: {type(images[0])}")
