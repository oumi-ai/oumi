"""Porting the Alpaca dataset with Oumi.

For more info see:
    (1) https://github.com/tatsu-lab/stanford_alpaca
    (2) https://github.com/gururise/AlpacaDataCleaned
"""

from typing import Union, cast

import pandas as pd

from oumi.core.datasets import BaseSftDataset
from oumi.core.registry import register_dataset
from oumi.core.types.conversation import Conversation, Message, Role

@register_dataset("RUC-AIBOX/long_form_thought_data_5k")
@register_dataset("O1-OPEN/OpenO1-SFT")
class PromptResponseDataset(BaseSftDataset):

    default_dataset = "O1-OPEN/OpenO1-SFT"

    def __init__(
        self,
        *,
        prompt_column: str = "instruction",
        response_column: str = "output",
        **kwargs,
    ) -> None:
        """Initializes a new instance of the PromptResponse class."""
        self.prompt_column = prompt_column
        self.response_column = response_column

        super().__init__(**kwargs)

    def transform_conversation(self, example: Union[dict, pd.Series]) -> Conversation:
        """Preprocesses the inputs of the example and returns a dictionary.

        Args:
            example (dict or Pandas Series): An example containing `input` (optional),
                `instruction`, and `output` entries.

        Returns:
            dict: The input example converted to Alpaca dictionary format.

        """
        messages = []

        # Use default Alpaca user prompt template
        user_prompt = cast(str, example[self.prompt_column])
        model_output = cast(str, example[self.response_column])

        # Create message list
        messages.append(Message(role=Role.USER, content=user_prompt))
        messages.append(Message(role=Role.ASSISTANT, content=model_output))

        return Conversation(messages=messages)
