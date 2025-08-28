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

from typing import Any, Optional, Union, cast

from PIL import Image
from typing_extensions import override

from oumi.builders.processors import build_processor
from oumi.core.configs.internal.internal_model_config import (
    InternalFeatureFirstDimAction,
    InternalModelConfig,
)
from oumi.core.configs.internal.supported_models import (
    find_internal_model_config_using_model_name,
    get_default_vlm_model_config,
)
from oumi.core.datasets.base_dpo_dataset import BaseDpoDataset
from oumi.core.tokenizers.base_tokenizer import BaseTokenizer
from oumi.core.types.conversation import ContentItem, Role, Type
from oumi.utils.conversation_utils import load_pil_image_from_content_item

_PROMPT_KEY = "prompt"
_CHOSEN_KEY = "chosen"
_REJECTED_KEY = "rejected"
_IMAGES_KEY = "images"


class VisionLanguageDpoDataset(BaseDpoDataset):
    """Dataset for vision-language DPO (Direct Preference Optimization) models.

    This class extends BaseDpoDataset to provide functionality specific to
    vision-language preference optimization tasks. It handles the processing of
    both image and text data for preference learning.

    The dataset expects data in the formats::

        {
            "prompt": "What's in this image?",
            "images": ["path/to/image.jpg", ...],  # Optional image paths/URLs
            "chosen": [{"role": "assistant", "content": "I see a cat"}],
            "rejected": [{"role": "assistant", "content": "I see a dog"}]
        }

        OR

        {
            "prompt": "What's in this image?",aths/URLs
            "images": ["path/to/image.jpg", ...],
            "chosen": "preferred response",
            "rejected": "rejected response"
        }
    """

    def __init__(
        self,
        *,
        dataset_name: Optional[str] = None,
        dataset_path: Optional[str] = None,
        split: Optional[str] = None,
        tokenizer: Optional[BaseTokenizer] = None,
        return_tensors: bool = False,
        processor: Optional[Any] = None,
        processor_name: Optional[str] = None,
        trust_remote_code: bool = False,
        processor_kwargs: Optional[dict[str, Any]] = None,
        max_size: Optional[int] = None,
        prompt_key: str = _PROMPT_KEY,
        chosen_key: str = _CHOSEN_KEY,
        rejected_key: str = _REJECTED_KEY,
        images_key: str = _IMAGES_KEY,
        **kwargs,
    ) -> None:
        """Initializes a new instance of the VisionLanguageDpoDataset class.

        The dataset will return dictionaries containing formatted preference data
        ready for DPO training with chat templates applied.

        Args:
            processor: The vision-language processor for applying chat templates
                and processing images.
            tokenizer: The tokenizer for encoding text data.
            return_tensors: Whether to return tensors instead of strings.
            dataset_name: The name of the dataset.
            dataset_path: The path to the dataset.
            split: The split of the dataset.
            processor_name: The name of the processor to use.
            trust_remote_code: Whether to trust remote code.
            processor_kwargs: Additional keyword arguments to pass to the processor.
            max_size: The maximum size of the longest edge of the image in pixels.
            prompt_key: The key for the prompt.
            chosen_key: The key for the chosen.
            rejected_key: The key for the rejected.
            images_key: The key for the images.
            **kwargs: Additional keyword arguments to pass to the base class.
        """
        super().__init__(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            split=split,
            tokenizer=tokenizer,
            return_tensors=return_tensors,
            **kwargs,
        )

        # Build the processor.
        if processor is not None:
            self._processor = processor
        elif processor_name is not None and self._tokenizer is not None:
            self._processor = build_processor(
                processor_name,
                self._tokenizer,
                trust_remote_code=True,
                processor_kwargs=processor_kwargs,
            )
        else:
            raise ValueError("Processor is not set.")

        self._internal_model_config: InternalModelConfig = (
            find_internal_model_config_using_model_name(
                self._processor.processor_name, trust_remote_code=True
            )
            or get_default_vlm_model_config()
        )
        self._max_size = max_size
        self._input_prompt_key = prompt_key
        self._input_chosen_key = chosen_key
        self._input_rejected_key = rejected_key
        self._input_images_key = images_key

    @override
    def transform_preference(self, sample: dict) -> dict:
        """Transform a DPO sample to the format expected by DPO trainer.

        Transforms a raw DPO example into three Oumi Conversation objects.

        Args:
            sample (dict): A dictionary representing a single DPO preference example.

        Returns:
            Dict with prompt, chosen, and rejected conversations or features
        """
        # First, convert the prompt, chosen, and rejected to conversation dictionaries.
        prompt_chat = self._to_messages_list(sample[self._input_prompt_key], Role.USER)
        chosen_chat = self._to_messages_list(
            sample[self._input_chosen_key], Role.ASSISTANT
        )
        rejected_chat = self._to_messages_list(
            sample[self._input_rejected_key], Role.ASSISTANT
        )
        images = sample[self._input_images_key] or []
        images = [images] if isinstance(images, dict) else images

        # Load and resize the images.
        if images is not None:
            images = [self._resize_image(self._load_image(image)) for image in images]

        # Add the image turns to the prompt if not already present.
        if all(isinstance(turn["content"], str) for turn in prompt_chat):
            for image in images:
                prompt_chat.append(
                    {
                        "role": "user",
                        "content": [{"type": "image_bytes", "content": image}],
                    }
                )

        return {
            _PROMPT_KEY: prompt_chat,
            _CHOSEN_KEY: chosen_chat,
            _REJECTED_KEY: rejected_chat,
            _IMAGES_KEY: images,
        }

    def _load_image(self, image_path: Union[str, ContentItem, dict]) -> Image.Image:
        """Load images from the given paths."""
        if isinstance(image_path, str):
            content_type = (
                Type.IMAGE_URL if image_path.startswith("http") else Type.IMAGE_PATH
            )
            image = ContentItem(type=content_type, content=image_path)
        elif isinstance(image_path, dict):
            image = ContentItem(type=Type.IMAGE_BINARY, binary=image_path["bytes"])
        else:
            image = image_path

        return load_pil_image_from_content_item(image)

    def _resize_image(self, image: Image.Image) -> Image.Image:
        if self._processor is None:
            return image

        if self._max_size is not None:
            image.thumbnail((self._max_size, self._max_size))

        return image

    def _drop_first_dim_if_needed(self, feature_name: str, value: Any) -> Any:
        """Drop the first dimension of the features."""
        feature_spec = self._internal_model_config.model_input_features.get(
            feature_name,
        )

        if feature_spec is None:
            action = InternalFeatureFirstDimAction.DROP_IF_DUMMY
        else:
            action = feature_spec.first_dim_action

        if action == InternalFeatureFirstDimAction.DROP_ALWAYS:
            return value[0]
        elif action == InternalFeatureFirstDimAction.DROP_IF_DUMMY:
            if len(value) == 1:
                return value[0]
            else:
                return value
        return value

    @override
    def _process_sample(
        self,
        features,
    ):
        """Process a row of the dataset."""
        if self._tokenizer is None or self._processor is None:
            raise ValueError(
                "Tokenizer and processor are required to process a sample."
            )

        # Apply the chat template to the prompt.
        prompt = self._tokenizer.apply_chat_template(features["prompt"], tokenize=False)
        prompt = cast(str, prompt)

        # Apply the chat template to the chosen and rejected turns.
        # To get only the completion part, we tokenizer the prompt + chosen/rejected
        # and then remove the prompt prefix.
        prompt_chosen = self._tokenizer.apply_chat_template(
            features["prompt"] + features["chosen"], tokenize=False
        )
        prompt_chosen = cast(str, prompt_chosen)
        chosen = prompt_chosen[len(prompt) :]

        prompt_rejected = self._tokenizer.apply_chat_template(
            features["prompt"] + features["rejected"], tokenize=False
        )
        prompt_rejected = cast(str, prompt_rejected)
        rejected = prompt_rejected[len(prompt) :]

        # Tokenize the prompt, chosen, and rejected turns.
        processed_features = self._processor(images=features["images"], text=[prompt])

        prompt_input_ids = self._drop_first_dim_if_needed(
            "input_ids", processed_features["input_ids"]
        )

        chosen_input_ids = self._tokenizer(chosen, add_special_tokens=False)[
            "input_ids"
        ]
        chosen_input_ids = cast(list[int], chosen_input_ids)
        rejected_input_ids = self._tokenizer(rejected, add_special_tokens=False)[
            "input_ids"
        ]
        rejected_input_ids = cast(list[int], rejected_input_ids)
        chosen_input_ids = chosen_input_ids + [self._tokenizer.eos_token_id]
        rejected_input_ids = rejected_input_ids + [self._tokenizer.eos_token_id]

        output = {
            "prompt_input_ids": prompt_input_ids,
            "chosen_input_ids": chosen_input_ids,
            "rejected_input_ids": rejected_input_ids,
        }

        # Drop the first dimension of the features if needed.
        if "pixel_values" in processed_features:
            output["pixel_values"] = self._drop_first_dim_if_needed(
                "pixel_values", processed_features["pixel_values"]
            )
        if "pixel_attention_mask" in processed_features:
            output["pixel_attention_mask"] = self._drop_first_dim_if_needed(
                "pixel_attention_mask", processed_features["pixel_attention_mask"]
            )
        if "image_sizes" in processed_features:
            output["image_sizes"] = self._drop_first_dim_if_needed(
                "image_sizes", processed_features["image_sizes"]
            )

        return output

    @override
    def transform(self, sample: dict) -> dict:
        """Transform the sample to the Oumi format."""
        return self._process_sample(self.transform_preference(sample))
