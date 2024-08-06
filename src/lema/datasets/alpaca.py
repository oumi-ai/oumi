from typing import Callable, Dict, Union, cast

import pandas as pd

from lema.core.datasets.base_dataset import BaseLMSftDataset
from lema.core.registry import register_dataset
from lema.core.types.base_tokenizer import BaseTokenizer
from lema.core.types.turn import Conversation, Message, Role
from lema.datasets.common import apply_chat_template


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


#
# Deprecated
#
def _convert_to_lema_format(example: dict) -> dict:
    """Converts the input example to the LeMa format."""
    messages = []
    metadata = {}

    # Use default alpaca system prompt
    system_prompt = (
        "Below is an instruction that describes a task, "
        "paired with an input that provides further context. "
        "Write a response that appropriately completes the request."
    )

    # Use default aplaca user prompt template
    if example.get("input") is not None and len(example["input"]) > 0:
        # This example has both an instruction and a user input.
        user_prompt = """{instruction}\n\n### Input:\n{input}""".format(
            instruction=example["instruction"], input=example["input"]
        )
    else:
        user_prompt = example["instruction"]

    # Create message list
    messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    messages.append({"role": "assistant", "content": example["output"]})

    return {"messages": messages, "metadata": metadata}


def alpaca_preprocessing_fn(
    tokenizer: BaseTokenizer,
) -> Callable[..., Dict]:
    """Builds a preprocessing function for the Alpaca dataset.

    Dataset: https://huggingface.co/datasets/tatsu-lab/alpaca
    """

    def prompt_generation_fn(sample) -> dict:
        sample = _convert_to_lema_format(sample)
        results = apply_chat_template(sample, tokenizer=tokenizer, task="sft")
        return results

    return prompt_generation_fn
