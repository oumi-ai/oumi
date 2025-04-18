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

from abc import ABC, abstractmethod
from typing import Any, Optional

from typing_extensions import override

from oumi.core.datasets import BaseSftDataset
from oumi.core.feature_generators import VisionLanguageConversationFeatureGenerator
from oumi.core.processors.base_processor import BaseProcessor
from oumi.core.tokenizers.base_tokenizer import BaseTokenizer
from oumi.core.types.conversation import (
    Conversation,
)
from oumi.utils.conversation_utils import remove_excessive_images_from_conversation


class VisionLanguageSftDataset(BaseSftDataset, ABC):
    """Abstract dataset for vision-language models.

    This class extends BaseSftDataset to provide functionality specific to
    vision-language tasks. It handles the processing of both image and text data.

    Note:
        This dataset is designed to work with models that can process both
        image and text inputs simultaneously, such as CLIP, BLIP, or other
        multimodal architectures.

    Example:
        >>> from oumi.builders import build_processor, build_tokenizer
        >>> from oumi.core.configs import ModelParams
        >>> from oumi.core.types.conversation import Conversation
        >>> from oumi.core.datasets import VisionLanguageSftDataset
        >>> class MyVisionLanguageSftDataset(VisionLanguageSftDataset):
        ...     def transform_conversation(self, example: dict):
        ...         # Implement the abstract method
        ...         # Convert the raw example into a Conversation object
        ...         pass
        >>> tokenizer = build_tokenizer(
        ...     ModelParams(model_name="Qwen/Qwen2-1.5B-Instruct")
        ... )
        >>> dataset = MyVisionLanguageSftDataset( # doctest: +SKIP
        ...     tokenizer=tokenizer,
        ...     processor_name="openai/clip-vit-base-patch32",
        ...     dataset_name="coco_captions",
        ...     split="train"
        ... )
        >>> sample = next(iter(dataset))  # doctest: +SKIP
        >>> print(sample.keys()) # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        return_conversations: bool = False,
        tokenizer: Optional[BaseTokenizer] = None,
        processor: Optional[BaseProcessor] = None,
        processor_name: Optional[str] = None,
        processor_kwargs: Optional[dict[str, Any]] = None,
        limit: Optional[int] = None,
        trust_remote_code: bool = False,
        max_images: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Initializes a new instance of the VisionLanguageDataset class.

        If `return_conversations` is True, the dataset will return dictionaries
            containing just JSON-encoded `Conversation` objects:
            {"conversation_json": conversation.to_json()}
        Otherwise, the dataset will return dictionaries containing model inputs:
            {"input_ids": ..., "attention_mask": ..., "pixel_values": ...}

        Args:
            tokenizer: A tokenizer for encoding text data.
            processor: An optional processor object for generating features.
            processor_name: The name of the processor to use for feature generation.
            processor_kwargs: A dictionary of processor-specific parameters.
                These parameters are passed to the processor constructor.
                They can override model-specific parameters.
            limit: An optional limit on the number of examples to load.
            trust_remote_code: Whether to trust remote code execution for the processor.
            return_conversations: Whether to return raw `Conversation` objects.
            max_images: The maximum number of images per conversation.
                If the limit is exceeded, the first N images are kept.
                If None or negative, all images are kept intact.
                If 0, all images are dropped.
            **kwargs: Additional keyword arguments to pass to the base class.
        """
        super().__init__(tokenizer=tokenizer, **kwargs)

        self._max_images = max_images

        self._feature_generator = (
            None
            if return_conversations
            else VisionLanguageConversationFeatureGenerator(
                tokenizer=tokenizer,
                processor=processor,
                processor_name=processor_name,
                processor_kwargs=processor_kwargs,
                trust_remote_code=trust_remote_code,
                return_tensors=self._return_tensors,
            )
        )

        if limit is not None:
            # TODO: this should be removed when we switch to datapipes.
            # Right now, we have to iterate over the whole dataset at init time,
            # Which takes way to long.
            self._data = self._data.head(limit)

    @abstractmethod
    def transform_conversation(self, example: dict) -> Conversation:
        """Transforms a raw example into an Oumi Conversation object.

        Args:
            example (dict): A dictionary representing a single conversation example.

        Returns:
            Conversation: A Conversation object representing the conversation.
        """
        raise NotImplementedError

    @override
    def transform(self, sample: dict) -> dict:
        """Transforms an Oumi conversation into a dictionary of inputs for a model.

        Args:
            sample (dict): A dictionary representing a single conversation example.

        Returns:
            dict: A dictionary of inputs for a model.
        """
        conversation = self.transform_conversation(sample)
        if self._max_images is not None and self._max_images >= 0:
            conversation = remove_excessive_images_from_conversation(
                conversation, max_images=self._max_images
            )

        if self._feature_generator is None:
            # This is only compatible with `use_torchdata=True`
            # as HF loaders expect certain keys like `input_ids`.
            conversation_json = conversation.to_json()
            return {"conversation_json": conversation_json}

        return self._feature_generator.transform_conversation(conversation, None)
