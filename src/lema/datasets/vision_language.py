# ruff: noqa: D102
import io
from typing import Any, Optional, Union

import requests
from PIL import Image

from lema.core.datasets import BaseLMSftDataset
from lema.core.registry import register_dataset
from lema.core.types.turn import Conversation, Message, Role, Type


class VisionLanguageSftDataset(BaseLMSftDataset):
    def __init__(
        self,
        *,
        image_processor: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        processor: Optional[Any] = None,
        **kwargs,
    ) -> None:
        """Initializes a new instance of the VisionLanguageDataset class."""
        super().__init__(**kwargs)

        if processor and (image_processor or tokenizer):
            raise ValueError()

        self._processor = processor

        if self._processor is not None:
            self._tokenizer = self._processor.tokenizer
            self._image_processor = self._processor.image_processor
        else:
            self._image_processor = image_processor
            self._tokenizer = tokenizer

        self._data = self._load_data()

    def transform_conversation(self, example: dict) -> Conversation:
        raise NotImplementedError("Subclasses must implement this method")

    def transform(self, sample: dict) -> dict:
        if self._processor is None and (
            self._tokenizer is None or self._image_processor is None
        ):
            raise ValueError

        conversation = self.transform_conversation(sample)

        images = [turn for turn in conversation.messages if turn.is_image()]

        texts = []
        for turn in conversation.messages:
            if turn.is_text():
                texts.append(turn)
            elif turn.is_image():
                placeholder = {
                    "content": [{"type": "image"}],
                    "role": str(turn.role),
                }
                texts.append(placeholder)
            else:
                raise ValueError(f"Unsupported message type: {turn.type}")

        if self._processor is not None:
            images = [self._load_image(image) for image in images]
            text = self._processor.apply_chat_template(
                texts, add_generation_prompt=False
            )
            inputs = self._processor(
                images=images, text=[text], return_tensors="pt", padding=True
            )
        else:
            if len(images) > 0:
                image = images[0]  # only support a single image
                image_features = self.transform_image(image)
            else:
                image_features = {}

            # TODO: fix type ignore
            text_features = self.tokenize(texts)  # type: ignore
            inputs = {**text_features, **image_features}

        inputs["labels"] = inputs["input_ids"]
        return inputs

    def _load_image(self, image: Union[str, Message]) -> Image.Image:
        if self._image_processor is None:
            raise ValueError("Processor required for transform")

        if isinstance(image, str):
            image = Message(type=Type.IMAGE_PATH, content=image, role=Role.USER)

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
