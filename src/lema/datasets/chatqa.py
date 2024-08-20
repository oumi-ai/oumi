from typing import Callable, Dict, Optional, Union, cast

import datasets
import pandas as pd

from lema.core.datasets.base_dataset import BaseLMSftDataset
from lema.core.registry import register_dataset
from lema.core.tokenizers import BaseTokenizer
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
            # Note: `tatqa` is a combination of `tatqa-arithmetic` and `tatqa-others``
            # Here we use the same prompt as tqta-arithmetic for both as it's larger
            # (8k vs 3k) Samples.
            # Preferably, each dataset should be loaded separately instead of loading
            # the combined `tatqa` subset.
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

        # Step 1. Add system message. Most subsets contain one.
        system_message = self._get_system_message()
        if system_message:
            messages.append(Message(role=Role.SYSTEM, content=system_message))

        # Step 2. Add grounding context and system instruction
        has_context = raw_conversation.get("document") is not None
        if has_context:
            # Step 2.1. If the sample has a context, we add a system prompt
            # to only use information from the context to answer the question
            context_message = (
                "Only use the information from the user "
                "provided context to answer the question."
            )
            messages.append(Message(role=Role.SYSTEM, content=context_message))

            # Step 2.2. Add context document, wrapped in <context> tags
            # Note: This is not part of the original dataset
            # but is added to make the context more explicit.
            document = f"<context>{raw_conversation['document']}</document>"
            messages.append(Message(role=Role.USER, content=document))

        # Step 3. Add conversation history
        # Can contain one or multiple user/assistant turns.
        for message in raw_conversation["messages"]:
            messages.append(Message(role=message["role"], content=message["content"]))

        # Step 4. Add final assistant response, which is encoded differently
        # depending on the subset.
        if self.dataset_subset == "narrativeqa":
            # `narrativeqa` contains an array of arrays of strings
            # Note: All rows contain two answers.
            # We arbitrarily use the first one answer.
            response = raw_conversation["answers"][0][0]
        elif self.dataset_subset in ("squad1.1", "squad2.0"):
            # `squad1.1` and `squad2.0` contain a list of dicts
            # All rows contain a single answer.
            response = raw_conversation["answers"][0]["text"]
        else:
            # All other subsets contain a list of strings
            # All rows contain a single answer.
            response = raw_conversation["answers"][0]
        messages.append({"role": Role.ASSISTANT, "content": response})

        return Conversation(messages=messages)


@register_dataset("nvidia/ChatQA-Training-Data", subset="tatqa-arithmetic")
@register_dataset("nvidia/ChatQA-Training-Data", subset="tatqa-others")
class ChatqaTatqaDataset(ChatqaDataset):
    """ChatQA Subclass to handle tatqa subsets.

    The tatqa subsets require loading a specific file from
    the dataset repository, thus requiring us to override the
    default loading behavior.
    """

    default_subset = "tatqa-arithmetic"

    def _load_hf_hub_dataset(self, path: str) -> pd.DataFrame:
        if self.dataset_subset == "tatqa-arithmetic":
            filename = "tatqa/train_arithmetic.json"
        else:
            filename = "tatqa/train_others.json"

        if self.split is not None and self.split != "train":
            raise ValueError("Only the `train` split is supported for this dataset.")

        dataset = datasets.load_dataset(
            self.dataset_name_or_path, data_files={"train": filename}
        )
        dataset = cast(datasets.DatasetDict, dataset)
        return cast(pd.DataFrame, dataset["train"].to_pandas())


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
    tokenizer: BaseTokenizer,
) -> Callable[..., Dict]:
    """Builds a preprocessing function for a TRL SFT (chat) trainer."""

    def prompt_generation_fn(sample) -> dict:
        sample = _convert_to_lema_format(sample)
        results = apply_chat_template(sample, tokenizer=tokenizer, task="sft")
        return results

    return prompt_generation_fn
