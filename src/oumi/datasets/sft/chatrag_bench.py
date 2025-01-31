# Copyright 2025 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Union

import pandas as pd
from typing_extensions import override

from oumi.core.datasets import BaseSftDataset
from oumi.core.registry import register_dataset
from oumi.core.types.conversation import Conversation, Message, Role


@register_dataset("nvidia/ChatRAG-Bench")
class ChatRAGBenchDataset(BaseSftDataset):
    default_dataset: str = "nvidia/ChatRAG-Bench"
    default_system_message: str = (
        "This is a chat between a user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's "
        "questions based on the context. The assistant should also indicate when "
        "the answer cannot be found in the context."
    )
    default_subset: str = "doc2dial"

    def __init__(
        self,
        *,
        split: str = "test",
        task: str = "generation",
        subset: Optional[str] = None,
        num_context_docs: int = 5,
        **kwargs,
    ) -> None:
        """Initialize the ChatRag dataset.

        Args:
            split: The split of the dataset to use. Defaults to "test".
            num_context_docs: The number of context documents to include in each
            example.
            subset: The subset of the dataset to use. Defaults to None.
            task: The task for which the dataset is intended. Defaults to "generation".
            **kwargs: Additional keyword arguments to be passed to the base class.
        """
        self.num_context_docs = num_context_docs

        subset = subset or self.default_subset

        # This dataset is for evaluation only and does not contain a training split.
        if split != "test":
            raise ValueError(
                f"This dataset only supports the 'test' split. Got: {split}"
            )

        if task != "generation":
            raise ValueError("This dataset can only be used for evaluation tasks")

        # Get the test split name for this subset, which may be different
        # from the Oumi user facing split.
        internal_split = self._get_test_dataset_split(subset)

        super().__init__(
            split=internal_split,
            subset=subset,
            task=task,
            **kwargs,
        )

    def _get_test_dataset_split(self, subset: str) -> str:
        # Most subset use the "test" split, except these three
        # Note: these datasets all have a single split (test or dev)
        # We consider them all to be test datasets.
        if subset in ("coqa", "inscit", "topiocqa"):
            return "dev"
        return "test"

    def _get_instruction(self) -> Optional[str]:
        """Get an appropriate instruction for each dataset subset."""
        subset_instructions = {
            "doc2dial": "Please give a full and complete answer for the question.",
            "quac": "Please give a full and complete answer for the question.",
            "qrecc": "Please give a full and complete answer for the question.",
            "coqa": (
                "Answer the following question with a short span. "
                "The answer needs to be just in a few words."
            ),
            "doqa": "Please give a full and complete answer for the question.",
            "convfinqa": "Please give a full and complete answer for the question.",
            "sqa": "Answer the following question with one or a list of items.",
            "topiocqa": (
                "Answer the following question with a short span, "
                "or a full and complete answer."
            ),
            "hybridial": "Please give a full and complete answer for the question.",
            "inscit": "Please give a full and complete answer for the question.",
        }

        if self.dataset_subset is None:
            raise ValueError("The dataset subset must be specified.")

        return subset_instructions.get(self.dataset_subset)

    def _format_context_document(self, doc: dict) -> str:
        # Not all docs have titles
        if doc["title"] is not None:
            return f"title: {doc['title']}, source: {doc['text']}"
        else:
            return f"source: {doc['text']}"

    @override
    def transform_conversation(self, example: Union[dict, pd.Series]) -> Conversation:
        """Transforms a given example into a Conversation object.

        Args:
            example (Union[dict, pd.Series]): The example to transform.

        Returns:
            Conversation: The transformed Conversation object.
        """
        messages = []

        # Add system message
        messages.append(Message(role=Role.SYSTEM, content=self.default_system_message))

        # Add grounding context
        grounding_documents = example["ctxs"][: self.num_context_docs]
        grounding_context = "\n\n".join(
            [self._format_context_document(doc) for doc in grounding_documents]
        )
        messages.append(Message(role=Role.USER, content="\n\n" + grounding_context))

        # Add conversation history
        added_instruction = False
        for turn in example["messages"]:
            content = turn["content"]

            if turn["role"] == "user" and not added_instruction:
                # Add user instruction to the first user turn
                content = f"{self._get_instruction()} {content}"
                added_instruction = True

            messages.append(Message(role=turn["role"], content=content))

        return Conversation(messages=messages)
