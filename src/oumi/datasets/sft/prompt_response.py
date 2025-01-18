"""Generic class for using HuggingFace datasets with input/output columns.

Allows users to specify the prompt and response columns at the config level.
"""

from typing import Union

import pandas as pd

from oumi.core.datasets import BaseSftDataset
from oumi.core.registry import register_dataset
from oumi.core.types.conversation import Conversation, Message, Role


@register_dataset("PromptResponseDataset")
class PromptResponseDataset(BaseSftDataset):
    """Converts HuggingFace Datasets with input/output columns to Message format.

    Example:
        dataset = PromptResponseDataset(hf_dataset_path="O1-OPEN/OpenO1-SFT",
        prompt_column="instruction",
        response_column="output")
    """

    default_dataset = "O1-OPEN/OpenO1-SFT"

    def __init__(
        self,
        *,
        hf_dataset_path: str = "O1-OPEN/OpenO1-SFT",
        prompt_column: str = "instruction",
        response_column: str = "output",
        **kwargs,
    ) -> None:
        """Initializes a new instance of the PromptResponseDataset class."""
        self.prompt_column = prompt_column
        self.response_column = response_column
        kwargs["dataset_name"] = hf_dataset_path
        super().__init__(**kwargs)

    def transform_conversation(self, example: Union[dict, pd.Series]) -> Conversation:
        """Preprocesses the inputs of the example and returns a dictionary.

        Args:
            example (dict or Pandas Series): An example containing `input` (optional),
                `instruction`, and `output` entries.

        Returns:
            dict: The input example converted to messages dictionary format.

        """
        messages = []

        user_prompt = str(example[self.prompt_column])
        model_output = str(example[self.response_column])

        # Create message list
        messages.append(Message(role=Role.USER, content=user_prompt))
        messages.append(Message(role=Role.ASSISTANT, content=model_output))

        return Conversation(messages=messages)
