from typing import Dict, Union

import pandas as pd

from oumi.core.datasets import BaseLMSftDataset
from oumi.core.registry import register_dataset
from oumi.core.types.conversation import Conversation, Message, Role


@register_dataset("argilla/magpie-ultra-v0.1")
class ArgillaMagpieUltraDataset(BaseLMSftDataset):
    """Dataset class for the argilla/magpie-ultra-v0.1 dataset."""

    default_dataset = "argilla/magpie-ultra-v0.1"

    def transform_conversation(self, example: Union[Dict, pd.Series]) -> Conversation:
        """Transform a dataset example into a Conversation object."""
        instruction = example.get("instruction", "")
        response = example.get("response", "")

        messages = [
            Message(role=Role.USER, content=instruction),
            Message(role=Role.ASSISTANT, content=response),
        ]

        return Conversation(messages=messages)


@register_dataset("Magpie-Align/Llama-3-Magpie-Pro-1M-v0.1")
@register_dataset("Magpie-Align/Magpie-Pro-300K-Filtered")
class MagpieProDataset(BaseLMSftDataset):
    """Dataset class for the Magpie-Align/Llama-3-Magpie-Pro-1M-v0.1 dataset."""

    default_dataset = "Magpie-Align/Llama-3-Magpie-Pro-1M-v0.1"

    def transform_conversation(self, example: Union[Dict, pd.Series]) -> Conversation:
        """Transform a dataset example into a Conversation object."""
        conversation = example.get("conversations")

        if conversation is None:
            raise ValueError("Conversation is None")

        messages = []
        for message in conversation:
            if message["from"] == "human":
                role = Role.USER
            elif message["from"] == "gpt":
                role = Role.ASSISTANT
            else:
                raise ValueError(f"Unknown role: {message['from']}")
            content = message.get("value", "")
            messages.append(Message(role=role, content=content))
        return Conversation(messages=messages)
