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

from typing import Any, Optional, Union

from PIL import Image
from typing_extensions import override

from oumi.core.datasets.base_dpo_dataset import BaseExperimentalDpoDataset
from oumi.core.tokenizers.base_tokenizer import BaseTokenizer
from oumi.core.types.conversation import ContentItem, Type
from oumi.utils.conversation_utils import load_pil_image_from_content_item

_PROMPT_KEY = "prompt"
_CHOSEN_KEY = "chosen"
_REJECTED_KEY = "rejected"
_IMAGES_KEY = "images"


class VisionLanguageDpoDataset(BaseExperimentalDpoDataset):
    """Dataset for vision-language DPO (Direct Preference Optimization) models.

    This class extends BaseExperimentalDpoDataset to provide functionality specific to
    vision-language preference optimization tasks. It handles the processing of
    both image and text data for preference learning.

    The dataset expects data in the format:
    {
        "prompt": "What's in this image?",
        "images": ["path/to/image.jpg", ...],  # Optional image paths/URLs
        "chosen": [{"role": "assistant", "content": "I see a cat"}],
        "rejected": [{"role": "assistant", "content": "I see a dog"}]
    }

    Example:
        >>> from oumi.builders import build_processor, build_tokenizer
        >>> from oumi.core.configs import ModelParams
        >>> from oumi.core.datasets import VisionLanguageDpoDataset
        >>> class MyVisionLanguageDpoDataset(VisionLanguageDpoDataset):
        ...     def transform_preference(self, example: dict):
        ...         # Implement the abstract method
        ...         # Convert the raw example into preference conversations
        ...         pass
        >>> tokenizer = build_tokenizer(
        ...     ModelParams(model_name="llava-hf/llava-1.5-7b-hf")
        ... )
        >>> dataset = MyVisionLanguageDpoDataset( # doctest: +SKIP
        ...     tokenizer=tokenizer,
        ...     processor_name="llava-hf/llava-1.5-7b-hf",
        ...     dataset_name="my_vision_dpo_dataset",
        ...     split="train"
        ... )
        >>> sample = next(iter(dataset))  # doctest: +SKIP
        >>> print(sample.keys()) # doctest: +SKIP
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
        self._processor = processor

    @override
    def transform_preference(self, sample: dict) -> dict:
        """Transform a DPO sample to the format expected by DPO trainer.

        Args:
            sample: Raw preference data sample

        Returns:
            Dict with prompt, chosen, and rejected conversations or features

        Transforms a raw DPO example into three Oumi Conversation objects.

        Args:
            example (dict): A dictionary representing a single DPO preference example.
                Expected format:
                {
                    "prompt": "What's in this image?",
                    "images": ["path/to/image.jpg", ...],  # Optional
                    "chosen": [{"role": "assistant", "content": "preferred response"}],
                    "rejected": [{"role": "assistant", "content": "rejected response"}]
                }

        Returns:
            Dict with prompt, chosen, and rejected conversations or features
        """
        prompt = sample[_PROMPT_KEY]
        chosen_chat = sample[_CHOSEN_KEY]
        rejected_chat = sample[_REJECTED_KEY]
        images = sample[_IMAGES_KEY] or []

        if images is not None:
            images = [self._resize_image(self._load_image(image)) for image in images]

        # Only use the last message of the chosen and rejected chats.
        # TODO: add support for conversation format
        chosen_chat_response = self._extract_from_chat_format(chosen_chat)
        rejected_chat_response = self._extract_from_chat_format(rejected_chat)

        return {
            _PROMPT_KEY: prompt,
            _CHOSEN_KEY: chosen_chat_response,
            _REJECTED_KEY: rejected_chat_response,
            _IMAGES_KEY: images,
        }

    def _load_image(self, image_path: Union[str, ContentItem]) -> Image.Image:
        """Load images from the given paths."""
        if isinstance(image_path, str):
            content_type = (
                Type.IMAGE_URL if image_path.startswith("http") else Type.IMAGE_PATH
            )
            image = ContentItem(type=content_type, content=image_path)
        else:
            image = image_path

        return load_pil_image_from_content_item(image)

    def _resize_image(self, image: Image.Image) -> Image.Image:
        if self._processor is None:
            return image

        # If the processor has an image processor, resize the image to the
        # longest edge of the image processor.
        if hasattr(self._processor, "image_processor") and hasattr(
            self._processor.image_processor, "size"
        ):
            max_size = self._processor.image_processor.size["longest_edge"]

        image.thumbnail((max_size, max_size))
        return image
