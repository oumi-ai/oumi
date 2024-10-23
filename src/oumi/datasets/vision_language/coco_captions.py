from typing_extensions import override

from oumi.core.datasets import VisionLanguageSftDataset
from oumi.core.registry import register_dataset
from oumi.core.types.conversation import Conversation, Message, Role, Type

_COCO_COLUMN_SENTENCES = "sentences"
_COCO_COLUMN_RAW = "raw"
_COCO_COLUMN_IMAGE = "image"
_COCO_COLUMN_PATH = "path"
_COCO_COLUMN_BYTES = "bytes"


@register_dataset("coco_captions")
class COCOCaptionsDataset(VisionLanguageSftDataset):
    default_dataset = "HuggingFaceM4/COCO"
    default_prompt = "Describe this image:"

    @override
    def transform_conversation(self, example: dict) -> Conversation:
        """Transform a single conversation example into a Conversation object."""
        input_text = self.default_prompt

        for required_key in (_COCO_COLUMN_SENTENCES, _COCO_COLUMN_IMAGE):
            if required_key not in example:
                raise ValueError(
                    "Training example doesn't contain '{required_key}' key. "
                    f"Available keys: {example.keys()}."
                )

        if _COCO_COLUMN_RAW not in example[_COCO_COLUMN_SENTENCES]:
            raise ValueError(
                "Training example doesn't contain 'sentences.raw' key. Available keys "
                f"under 'sentences.': {example[_COCO_COLUMN_SENTENCES].keys()}."
            )
        output_text = example[_COCO_COLUMN_SENTENCES][_COCO_COLUMN_RAW]

        messages = []

        if _COCO_COLUMN_BYTES in example[_COCO_COLUMN_IMAGE]:
            messages.append(
                Message(
                    role=Role.USER,
                    binary=example[_COCO_COLUMN_IMAGE][_COCO_COLUMN_BYTES],
                    type=Type.IMAGE_BINARY,
                )
            )
        elif _COCO_COLUMN_PATH in example[_COCO_COLUMN_IMAGE]:
            messages.append(
                Message(
                    role=Role.USER,
                    content=example[_COCO_COLUMN_IMAGE][_COCO_COLUMN_PATH],
                    type=Type.IMAGE_PATH,
                )
            )
        else:
            raise ValueError(
                "Training example contains none of required keys: "
                "'image.bytes', 'image.path'. "
                f"Available keys under 'image.': {example[_COCO_COLUMN_IMAGE].keys()}."
            )

        messages.append(Message(role=Role.USER, content=input_text))
        messages.append(Message(role=Role.ASSISTANT, content=output_text))

        return Conversation(messages=messages)
