from oumi.core.datasets import VisionLanguageSftDataset
from oumi.core.registry import register_dataset
from oumi.core.types.turn import Conversation, Message, Role


# @register_dataset("coco_captions")
@register_dataset("/home/user/data/coco_captions/train")
class COCOCaptionsDataset(VisionLanguageSftDataset):
    # default_dataset = "HuggingFaceM4/COCO"
    default_dataset = "/home/user/data/coco_captions/test"
    default_prompt = "Describe this image:"

    def transform_conversation(self, example: dict) -> Conversation:
        """Transform a single conversation example into a Conversation object."""
        input_text = self.default_prompt
        output_text = example["sentences"]["raw"]

        messages = [Message(role=Role.USER, content=input_text)]

        # Message(role=Role.USER,
        # content=example["image"]["path"], type=Type.IMAGE_PATH)
        messages.append(Message(role=Role.ASSISTANT, content=output_text))

        return Conversation(messages=messages)
