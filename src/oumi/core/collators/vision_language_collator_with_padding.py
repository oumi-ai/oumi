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
from oumi.core.feature_generators import (
    FeatureGeneratorOptions,
    VisionLanguageConversationFeatureGenerator,
)
from oumi.core.tokenizers.base_tokenizer import BaseTokenizer
from oumi.core.types import Conversation
from oumi.utils.torch_utils import pad_to_max_dim_and_stack

_PIXEL_VALUES_KEY = "pixel_values"


class VisionLanguageCollatorWithPadding:
    def __init__(
        self,
        tokenizer: BaseTokenizer,
        *,
        max_length: Optional[int],
        truncation: bool = False,
        label_ignore_index: Optional[int] = None,
        allow_multi_image_inputs: bool = True,
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
        """
        self._conversation_feature_generator: Optional[
            VisionLanguageConversationFeatureGenerator
        ] = None
        if True:
            self._conversation_feature_generator = (
                VisionLanguageConversationFeatureGenerator(
                    tokenizer=tokenizer,
                    processor=None,
                    processor_name="Qwen/Qwen2-VL-2B-Instruct",
                    trust_remote_code=True,
                    return_tensors="pt",
                )
            )

        self._allow_multi_image_inputs = allow_multi_image_inputs
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
        """Custom collator for multi-modal vision-language training.

        Args:
            batch: List of batch items.

        Returns:
            Dict[str, torch.Tensor]: Processed batch.
        """
        batch_size = len(batch)
        if batch_size <= 0:
            raise ValueError("Batch is empty")

        if self._conversation_feature_generator is None:
            print(
                f"self._conversation_feature_generator: "
                f"{self._conversation_feature_generator is not None}\n"
                f"batch[0]: {batch[0].keys()}"
            )
            return self._collate_batch(batch)

        assert self._conversation_feature_generator is not None

        conversations: list[Conversation] = []
        for idx in range(batch_size):
            if "conversation" not in batch[idx]:
                raise ValueError(
                    f"Example doesn't contain 'conversation' key. "
                    f"Example: {idx + 1} of {batch_size}. "
                    f"Available keys: {batch[idx].keys()}"
                )

            conversation_json = batch[idx]["conversation"]
            conversations.append(Conversation.from_json(conversation_json))
        assert len(conversations) == batch_size

        if True:
            updated_batch: list[dict] = []
            for conversation in conversations:
                updated_batch.append(
                    self._conversation_feature_generator.transform_conversation(
                        conversation,
                        options=None,
                    )
                )
            result1 = self._collate_batch(updated_batch)

        result2 = self._conversation_feature_generator.transform_conversations(
            conversations,
            FeatureGeneratorOptions(allow_feature_reshape=False),
        )
        for idx, res in enumerate([result1, result2]):
            res_shapes = {k: res[k].shape for k in sorted(res.keys())}
            print(f"result{idx + 1}: {res_shapes}")

        result = result2

        # TODO: Handle truncation.

        return result

    def _collate_batch(self, batch) -> dict[str, Any]:
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

        # Collate images.
        pixel_values = self._collate_images(images)

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

    def _collate_images(self, images) -> torch.Tensor:
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
