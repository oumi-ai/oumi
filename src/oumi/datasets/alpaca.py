from typing import Union, cast

import pandas as pd

from oumi.core.datasets import BaseLMSftDataset
from oumi.core.registry import register_dataset
from oumi.core.types.turn import Conversation, Message, Role


@register_dataset("yahma/alpaca-cleaned")
@register_dataset("tatsu-lab/alpaca")
class AlpacaDataset(BaseLMSftDataset):
    system_prompt = (
        "Below is an instruction that describes a task, "
        "paired with an input that provides further context. "
        "Write a response that appropriately completes the request."
    )

    default_dataset = "tatsu-lab/alpaca"

    def __init__(
        self,
        *,
        include_system_prompt: bool = True,
        **kwargs,
    ) -> None:
        """Initializes a new instance of the AlpacaDataset class."""
        self.include_system_prompt = include_system_prompt

        super().__init__(**kwargs)

    def transform_conversation(self, example: Union[dict, pd.Series]) -> Conversation:
        """Preprocesses the inputs of the example and returns a dictionary.

        Args:
            example (dict): The example containing the input and instruction.

        Returns:
            dict: The preprocessed inputs as a dictionary.

        """
        messages = []

        # Use default aplaca user prompt template
        if example.get("input") is not None and len(example["input"]) > 0:
            # This example has both an instruction and a user input.
            user_prompt = """{instruction}\n\n### Input:\n{input}""".format(
                instruction=example["instruction"], input=example["input"]
            )
        else:
            user_prompt = cast(str, example["instruction"])

        model_output = cast(str, example["output"])

        # Create message list
        if self.include_system_prompt:
            messages.append(Message(role=Role.SYSTEM, content=self.system_prompt))
        messages.append(Message(role=Role.USER, content=user_prompt))
        messages.append(Message(role=Role.ASSISTANT, content=model_output))

        return Conversation(messages=messages)
