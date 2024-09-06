# ruff: noqa: D102
import io
import os
from typing import Any, List, Optional, Tuple, Union

import pandas as pd
import requests
from PIL import Image
from transformers import AutoProcessor

from lema.core.datasets import BaseLMSftDataset
from lema.core.registry import register_dataset
from lema.core.types.turn import Conversation, Message, Role, Type
from lema.utils.logging import logger


class VisionLanguageSftDataset(BaseLMSftDataset):
    def __init__(
        self,
        *,
        processor: Optional[Any] = None,
        processor_name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Initializes a new instance of the VisionLanguageDataset class."""
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

        self._data = self._load_data()

        super().__init__(**kwargs)

    def transform_conversation(self, example: dict) -> Conversation:
        raise NotImplementedError("Subclasses must implement this method")

    def transform(self, sample: dict) -> dict:
        if self._processor is None:
            raise ValueError("Processor required for transform")

        conversation = self.transform_conversation(sample)

        if self._processor.chat_template is None:
            print("Using simple processor")
            image, prompt = self._prepare_simple_model(conversation)

            inputs = self._processor(
                images=image, text=prompt, return_tensors="pt", padding=True
            )
        else:
            images, prompt = self._prepare_instruct_model(conversation)

            inputs = self._processor(
                images=images, text=[prompt], return_tensors="pt", padding=True
            )

        inputs["labels"] = inputs["input_ids"]
        return inputs

    def _prepare_simple_model(
        self, conversation: Conversation
    ) -> Tuple[Image.Image, str]:
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

    def transform_images(self, messages):
        return [self.transform_image(message) for message in messages]

    def transform_image(self, message: Union[str, Message]):
        if self._image_processor is None:
            raise ValueError

        image_bin = self._load_image(message)
        features = self._image_processor(
            images=image_bin, return_tensors=self._return_tensors
        )
        return features


@register_dataset("coco_captions")
class COCOCaptionsDataset(VisionLanguageSftDataset):
    default_dataset = "HuggingFaceM4/COCO"

    def transform_conversation(self, example: dict) -> Conversation:
        input_text = "Describe this image:"
        output_text = example["sentences"]["raw"]

        messages = [
            Message(role=Role.USER, content=input_text),
            Message(
                role=Role.USER, content=example["image"]["path"], type=Type.IMAGE_PATH
            ),
            Message(role=Role.ASSISTANT, content=output_text),
        ]

        return Conversation(messages=messages)


@register_dataset("flickr30k")
class Flickr30kDataset(VisionLanguageSftDataset):
    default_dataset = "nlphuji/flickr30k"

    def transform_conversation(self, example: dict) -> Conversation:
        input_text = "Describe this image:"
        output_text = example["caption"][0]

        messages = [
            Message(role=Role.USER, content=input_text),
            Message(
                role=Role.USER,
                binary=example["image"]["bytes"],
                type=Type.IMAGE_BINARY,
            ),
            Message(role=Role.ASSISTANT, content=output_text),
        ]

        return Conversation(messages=messages)


@register_dataset("vqa_v2")
class VQAv2Dataset(VisionLanguageSftDataset):
    default_dataset = "vqa_v2"

    def transform_conversation(self, example: dict) -> Conversation:
        input_text = example["question"]
        output_text = example["answers"][0]["answer"]  # Using the first answer

        messages = [
            Message(role=Role.USER, content=input_text),
            Message(
                role=Role.USER, content=example["image"]["path"], type=Type.IMAGE_PATH
            ),
            Message(role=Role.ASSISTANT, content=output_text),
        ]

        return Conversation(messages=messages)


@register_dataset("vision_language_jsonl")
class JsonlinesDataset(VisionLanguageSftDataset):
    default_dataset = "custom"

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        data: Optional[list] = None,
        data_column: str = "messages",
        **kwargs,
    ):
        """Initializes a new instance of the JsonlinesDataset class."""
        self.data_column = data_column
        self.dataset_path = dataset_path

        if dataset_path is not None and data is not None:
            raise ValueError(
                "Either dataset_path or data must be provided, but not both"
            )

        if data is not None:
            self._data = pd.DataFrame({self.data_column: data})

        elif dataset_path is not None:
            if not os.path.isfile(dataset_path):
                raise ValueError(f"Dataset path does not exist: {dataset_path}")

            if not dataset_path.endswith(".jsonl"):
                raise ValueError("Dataset path must end with .jsonl")

            self._data = pd.read_json(dataset_path, lines=True)

            if self.data_column not in self._data.columns:
                raise ValueError(f"Data column {self.data_column} not found in dataset")
        else:
            raise ValueError("Dataset path or data must be provided")

        super().__init__(**kwargs)

    def _load_data(self):
        # no-op, data is already loaded in __init__
        return self._data

    def transform_conversation(self, example: dict) -> Conversation:
        return Conversation(messages=example["messages"])
