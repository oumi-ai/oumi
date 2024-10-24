from typing import Dict, Union

import pandas as pd

from oumi.core.datasets import BaseSftDataset
from oumi.core.registry import register_dataset
from oumi.core.types.conversation import Conversation, Message, Role


@register_dataset("CohereForAI/aya_dataset")
class AyaDataset(BaseSftDataset):
    """Dataset class for the CohereForAI/aya_dataset dataset."""

    default_dataset = "CohereForAI/aya_dataset"

    def transform_conversation(self, example: Union[Dict, pd.Series]) -> Conversation:
        """Transform a dataset example into a Conversation object."""
        question = example.get("inputs", "")
        answer = example.get("targets", "")

        messages = [
            Message(role=Role.USER, content=question),
            Message(role=Role.ASSISTANT, content=answer),
        ]

        return Conversation(messages=messages)
