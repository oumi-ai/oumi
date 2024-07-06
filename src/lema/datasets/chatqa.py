from typing import Callable, Dict, Optional, Union

import pandas as pd
from transformers import PreTrainedTokenizerBase

from lema.core.registry import register_dataset
from lema.core.types.base_dataset import BaseLMSftDataset
from lema.core.types.turn import Conversation, Message, Role
from lema.datasets.common import apply_chat_template


@register_dataset("nvidia/ChatQA-Training-Data")
class ChatqaDataset(BaseLMSftDataset):
    default_dataset = "nvidia/ChatQA-Training-Data"
    default_subset = "sft"

    def _get_system_message(self) -> Optional[str]:
        if self.dataset_subset == "sft":
            return None

        if self.dataset_subset == "synthetic_convqa":
            return "Please give a full and complete answer for the question."

        if self.dataset_subset in ("tatqa-arithmetic", "tatqa"):
            return (
                "Answer the following question with a number "
                "from context or the math arithmetic"
            )

        if self.dataset_subset == "tatqa-others":
            return (
                "Answer the following question with a short span, "
                "or a full and complete answer"
            )

        if self.dataset_subset in (
            "drop",
            "narrativeqa",
            "quoref",
            "ropes",
            "squad1.1",
            "squad2.0",
            "newsqa",
        ):
            return "Answer the following question with a short span."

        raise ValueError(f"Unknown dataset subset: {self.dataset_subset}")

    def transform_conversation(
        self, raw_conversation: Union[dict, pd.Series]
    ) -> Conversation:
        """Preprocesses the inputs of the example and returns a dictionary.

        ChatQA is a conversational question answering dataset.
        It contains 10 subsets. Some subsets contain grounding documents.

        See the dataset page for more information:
        https://huggingface.co/datasets/nvidia/ChatQA-Training-Data

        Args:
            raw_conversation: The raw conversation example.

        Returns:
            dict: The preprocessed inputs as a Lema conversation.
        """
        messages = []

        has_context = raw_conversation.get("document") is not None

        # Most subsets contain a system message
        system_message = self._get_system_message()
        if system_message:
            messages.append(Message(role=Role.SYSTEM, content=system_message))

        # If the sample has a context, we add a system prompt
        # to only use information from the context to answer the question
        if has_context:
            context_message = (
                "Only use the information from the user "
                "provided context to answer the question."
            )
            messages.append(Message(role=Role.SYSTEM, content=context_message))

            # Add context document, wrapped in <context> tags
            # Note: This is not part of the original dataset
            # but is added to make the context more explicit.
            document = f"<context>{raw_conversation['document']}</document>"
            messages.append(Message(role=Role.USER, content=document))

        # Add user question
        for message in raw_conversation["messages"]:
            messages.append(Message(role=message["role"], content=message["content"]))

        # Add assistant responses
        for response in raw_conversation["answers"]:
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
