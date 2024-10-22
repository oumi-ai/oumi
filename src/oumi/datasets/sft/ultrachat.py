from typing import Dict, Union

import pandas as pd

from oumi.core.datasets import BaseSftDataset
from oumi.core.registry import register_dataset
from oumi.core.types.conversation import Conversation, Message


@register_dataset("HuggingFaceH4/ultrachat_200k")
class UltrachatH4Dataset(BaseSftDataset):
    """Dataset class for the HuggingFaceH4/ultrachat_200k dataset."""

    default_dataset = "HuggingFaceH4/ultrachat_200k"

    def transform_conversation(self, example: Union[Dict, pd.Series]) -> Conversation:
        """Transform a dataset example into a Conversation object."""
        raw_messages = example.get("messages")
        if raw_messages is None:
            raise ValueError("Invalid messages")

        messages = [Message.model_validate(message) for message in raw_messages]
        return Conversation(messages=messages)
