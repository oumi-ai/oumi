"""Porting the Alpaca evaluation dataset with Oumi.

For more info see: https://github.com/tatsu-lab/alpaca_eval
"""

from typing import Union, cast

import pandas as pd

from oumi.core.datasets import BaseSftDataset
from oumi.core.registry import register_dataset
from oumi.core.types.conversation import Conversation, Message, Role


@register_dataset("tatsu-lab/alpaca_eval")
class AlpacaEvalDataset(BaseSftDataset):
    system_prompt_with_context = (
        "Below is an instruction that describes a task, "
        "paired with an input that provides further context. "
        "Write a response that appropriately completes the request."
    )

    system_prompt_without_context = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
    )

    default_dataset = "tatsu-lab/alpaca_eval"

    def __init__(
        self,
        *,
        include_system_prompt: bool = False,
        **kwargs,
    ) -> None:
        """Initializes a new instance of the AlpacaDataset class."""
        self.include_system_prompt = include_system_prompt

        super().__init__(**kwargs)

    def get_list_conversations(self) -> list[Conversation]:
        """Convert the dataset to a list of `Conversation`s for inference."""
        indexes = range(len(self))
        return [self.conversation(index) for index in indexes]

    def transform_conversation(self, example: Union[dict, pd.Series]) -> Conversation:
        """Preprocesses the inputs of the example and returns a dictionary.

        Args:
            example (dict): The example containing the input and instruction.

        Returns:
            dict: The preprocessed inputs as a dictionary.

        """
        messages = []

        # Use default Alpaca user prompt template.
        if example.get("input") is not None and len(example["input"]) > 0:
            # This example has both an instruction and a user input.
            user_prompt = """{instruction}\n\n### Input:\n{input}""".format(
                instruction=example["instruction"], input=example["input"]
            )
            system_prompt = self.system_prompt_with_context
        else:
            user_prompt = cast(str, example["instruction"])
            system_prompt = self.system_prompt_without_context

        # Create message list.
        if self.include_system_prompt:
            messages.append(Message(role=Role.SYSTEM, content=system_prompt))
        messages.append(Message(role=Role.USER, content=user_prompt))

        # Retain other fields as metadata.
        metadata_fields = set()
        if isinstance(example, pd.Series):
            metadata_fields = {str(i) for i in example.index}
        elif isinstance(example, dict):
            metadata_fields = {str(key) for key in example.keys()}
        metadata_fields = metadata_fields - {"instruction", "input"}
        metadata = {field: example[field] for field in metadata_fields}

        return Conversation(messages=messages, metadata=metadata)
