from typing import Callable, Dict, Union

import pandas as pd
from transformers import PreTrainedTokenizerBase

from lema.core.registry import register_dataset
from lema.core.types.base_dataset import BaseLMSftDataset
from lema.core.types.turn import Conversation, Message, Role
from lema.datasets.common import apply_chat_template


@register_dataset("nvidia/ChatQA-Training-Data")
class ChatQADataset(BaseLMSftDataset):
    default_dataset = "nvidia/ChatQA-Training-Data"

    def transform_conversation(self, example: Union[dict, pd.Series]) -> Conversation:
        """Preprocesses the inputs of the example and returns a dictionary.

        Args:
            example (dict): The example containing the input and instruction.

        Returns:
            dict: The preprocessed inputs as a dictionary.

        """
        messages = []

        for message in example["messages"]:
            messages.append(Message(role=message["role"], content=message["content"]))

        for response in example["answers"]:
            messages.append({"role": Role.ASSISTANT, "content": response})

        return Conversation(messages=messages)


#
# Deprecated
#
def _convert_to_lema_format(example: dict) -> dict:
    """Converts the input example to the LeMa format."""
    messages = example["messages"].copy()
    metadata = {}

    for response in example["answers"]:
        messages.append({"role": "assistant", "content": response})

    return {"messages": messages, "metadata": metadata}


def chatqa_preprocessor_fn(
    tokenizer: PreTrainedTokenizerBase,
) -> Callable[..., Dict]:
    """Builds a preprocessing function for a TRL SFT (chat) trainer."""

    def prompt_generation_fn(sample) -> dict:
        sample = _convert_to_lema_format(sample)
        results = apply_chat_template(sample, tokenizer=tokenizer, task="sft")
        return results

    return prompt_generation_fn
