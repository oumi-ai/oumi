from oumi.core.datasets import VisionLanguageSftDataset
from oumi.core.registry import register_dataset
from oumi.core.types.turn import Conversation, Message, Role, Type


@register_dataset("coco_captions")
class COCOCaptionsDataset(VisionLanguageSftDataset):
    default_dataset = "HuggingFaceM4/COCO"
    default_prompt = "Describe this image:"

    def transform_conversation(self, example: dict) -> Conversation:
        """Transform a single conversation example into a Conversation object."""
        input_text = self.default_prompt

        for required_key in ("sentences", "image"):
            if required_key not in example:
                raise ValueError(
                    "Training example doesn't contain '{required_key}' key. "
                    f"Available keys: {example.keys()}."
                )

        if "raw" not in example["sentences"]:
            raise ValueError(
                "Training example doesn't contain 'sentences.raw' key. "
                f"Available keys under 'sentences.': {example['sentences'].keys()}."
            )
        output_text = example["sentences"]["raw"]

        messages = [Message(role=Role.USER, content=input_text)]

        if "bytes" in example["image"]:
            messages.append(
                Message(
                    role=Role.USER,
                    binary=example["image"]["bytes"],
                    type=Type.IMAGE_BINARY,
                )
            )
        elif "path" in example["image"]:
            messages.append(
                Message(
                    role=Role.USER,
                    content=example["image"]["path"],
                    type=Type.IMAGE_PATH,
                )
            )
        else:
            raise ValueError(
                "Training example contains none of required keys: "
                "'image.bytes', 'image.path'. "
                f"Available keys under 'image.': {example['image'].keys()}."
            )

        messages.append(Message(role=Role.ASSISTANT, content=output_text))

        return Conversation(messages=messages)
