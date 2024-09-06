import io
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Union

import requests
import torch
from PIL import Image
from transformers import AutoProcessor

from lema.core.datasets import BaseLMSftDataset
from lema.core.types.turn import Conversation, Message, Role, Type
from lema.utils.logging import logger


class VisionLanguageSftDataset(BaseLMSftDataset, ABC):
    """Abstract dataset for vision-language models.

    This class extends BaseLMSftDataset to provide functionality specific to
    vision-language tasks. It handles the processing of both image and text data.

    Note:
        This dataset is designed to work with models that can process both
        image and text inputs simultaneously, such as CLIP, BLIP, or other
        multimodal architectures.

    Example:
        >>> class MyVisionLanguageSftDataset(VisionLanguageSftDataset):
        ...     def transform_conversation(self, example: dict) -> Conversation:
        ...         # Implement the abstract method
        ...         # Convert the raw example into a Conversation object
        ...         pass
        >>>
        >>> dataset = MyVisionLanguageSftDataset(
        ...     processor_name="openai/clip-vit-base-patch32",
        ...     dataset_name="coco_captions",
        ...     split="train"
        ... )
        >>> sample = next(iter(dataset))
        >>> print(sample.keys())
    """

    def __init__(
        self,
        *,
        processor: Optional[Any] = None,
        processor_name: Optional[str] = None,
        limit: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Initializes a new instance of the VisionLanguageDataset class."""
        super().__init__(**kwargs)

        if processor_name is not None and processor is not None:
            logger.warning(
                "Both processor and processor_name are provided. "
                "Ignoring processor_name: %s",
                processor_name,
            )

        if processor_name is not None and processor is None:
            processor = AutoProcessor.from_pretrained(processor_name)

        self._processor = processor
        self._processor_name = processor_name

        if self._processor is not None:
            self._tokenizer = self._processor.tokenizer
            self._image_processor = self._processor.image_processor
        else:
            self._tokenizer = None
            self._image_processor = None

        if limit is not None:
            # TODO: this should be removed when we switch to datapipes.
            # Right now, we have to iterate over the whole dataset at init time,
            # Which takes way to long.
            self._data = self._data.head(limit)

    @abstractmethod
    def transform_conversation(self, example: dict) -> Conversation:
        """Transforms a raw example into a lema Conversation object.

        Args:
            example (dict): A dictionary representing a single conversation example.

        Returns:
            Conversation: A Conversation object representing the conversation.
        """

    def transform_image(self, message: Union[str, Message]) -> torch.Tensor:
        """Transforms a single image from a message for debugging.

        Args:
            message (Union[str, Message]): A string representing the image path or a
                Message object.

        Returns:
            Image.Image: A PIL image.
        """
        if self._image_processor is None:
            raise ValueError

        image_bin = self._load_image(message)
        features = self._image_processor(
            images=image_bin, return_tensors=self._return_tensors
        )
        return features

    def transform(self, sample: dict) -> dict:
        """Transforms a lema conversation into a dictionary of inputs for a model.

        Args:
            sample (dict): A dictionary representing a single conversation example.

        Returns:
            dict: A dictionary of inputs for a model.
        """
        if self._processor is None:
            raise ValueError("Processor required for transform")

        conversation = self.transform_conversation(sample)

        if self._processor.chat_template is None:
            print("Using simple processor")
            image, prompt = self._prepare_simple_model(conversation)

            inputs = self._processor(
                images=image,
                text=prompt,
                return_tensors=self._return_tensors,
                padding=True,
            )
        else:
            images, prompt = self._prepare_instruct_model(conversation)

            inputs = self._processor(
                images=images,
                text=[prompt],
                return_tensors=self._return_tensors,
                padding=True,
            )

        inputs["labels"] = inputs["input_ids"]
        return inputs

    def _prepare_simple_model(
        self, conversation: Conversation
    ) -> Tuple[Image.Image, str]:
        """Prepares the images and prompt for a simple model.

        Simple models only use the last image and text turn in the conversation. They
        don't use the chat template, so the prompt is just the last text turn.
        """
        last_image_turn = [turn for turn in conversation.messages if turn.is_image()][
            -1
        ]
        last_text_turn = [turn for turn in conversation.messages if turn.is_text()][
            -1
        ].content or ""

        prompt = last_text_turn or ""
        image = self._load_image(last_image_turn)

        return image, prompt

    def _prepare_instruct_model(
        self, conversation: Conversation
    ) -> Tuple[List[Image.Image], str]:
        """Prepares the images and prompt for an instruct model.

        Instruct models use the chat template to generate the prompt, and can include
        multiple images and text turns.
        """
        if self._processor is None:
            raise ValueError("Processor is required for instruct model")

        # Generates the prompt using the chat template
        # including image placeholders for each image in the conversation
        texts = []
        for turn in conversation.messages:
            if turn.is_text():
                texts.append(turn)

            elif turn.is_image():
                image_placeholder = {
                    "content": [{"type": "image"}],
                    "role": str(turn.role),
                }
                texts.append(image_placeholder)
            else:
                raise ValueError(f"Unsupported message type: {turn.type}")

        text = self._processor.apply_chat_template(texts, add_generation_prompt=False)

        # Loads the images from the conversation
        images = [turn for turn in conversation.messages if turn.is_image()]
        images = [self._load_image(image) for image in images]

        return images, text

    def _load_image(self, image: Union[str, Message]) -> Image.Image:
        """Loads an image from a message.

        Args:
            image (Union[str, Message]): A string representing the image path or a
                Message object.

        Returns:
            Image.Image: A PIL image.
        """
        if self._image_processor is None:
            raise ValueError("Processor required for transform")

        if isinstance(image, str):
            image_type = Type.IMAGE_URL if image.startswith("http") else Type.IMAGE_PATH
            image = Message(type=image_type, content=image, role=Role.USER)

        if image.type == Type.IMAGE_PATH:
            if image.content is None:
                raise ValueError("Image path is None")
            image_bin = Image.open(image.content).convert("RGB")

        elif image.type == Type.IMAGE_URL:
            if image.content is None:
                raise ValueError("Image URL is None")
            response = requests.get(image.content, stream=True)
            response.raise_for_status()
            image_bin = Image.open(response.raw).convert("RGB")

        elif image.type == Type.IMAGE_BINARY:
            if image.binary is None:
                raise ValueError("Image binary is None")
            image_bin = Image.open(io.BytesIO(image.binary)).convert("RGB")

        else:
            raise ValueError(f"Unsupported image type: {image.type}")

        return image_bin
