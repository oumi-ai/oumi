"""TatQA dataset for GRPO training."""

import pandas as pd
from typing_extensions import override

from oumi.core.datasets.base_grpo_dataset import BaseExperimentalGrpoDataset
from oumi.core.registry import register_dataset
from oumi.core.types.conversation import Conversation


@register_dataset("tatqa_data")
class TatqaDataset(BaseExperimentalGrpoDataset):
    """Dataset class for the TatQA (Tabular Question Answering) dataset.

    TatQA is a dataset for question answering over financial tables.

    Expected data format (JSONL):
    {
        "conversation_id": "unique-id",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant..."},
            {"role": "user", "content": "Answer the question based on the following table and text..."},
            {"role": "assistant", "content": "<think>...</think><answer>...</answer>"}
        ],
        "metadata": {
            "ground_truth": "129,454"  # The canonical answer
        }
    }

    For GRPO training:
    - The prompt includes system + user messages (no assistant message)
    - Ground truth is extracted from metadata["ground_truth"]
    - GRPO will generate the assistant's response
    """

    default_dataset = "tatqa_dataset"

    @override
    def transform(self, sample: pd.Series) -> dict:
        """Transform the sample into GRPO format.

        Args:
            sample: Raw sample from the dataset

        Returns:
            Dictionary with:
            - prompt: List of messages (system + user only)
            - ground_truth: The canonical answer from metadata
            - conversation_id: Unique identifier for tracking
        """
        sample_dict = sample.to_dict()

        # Extract messages - remove assistant message for GRPO
        messages = sample_dict["messages"]
        prompt_messages = [msg for msg in messages if msg["role"] != "assistant"]

        # Extract ground truth from metadata
        ground_truth = sample_dict["metadata"]["ground_truth"]

        # Return GRPO-formatted sample
        output_dict = {
            "prompt": prompt_messages,
            "ground_truth": ground_truth,
            "conversation_id": sample_dict.get("conversation_id", ""),
        }
        return output_dict

    @override
    def transform_conversation(self, sample: pd.Series) -> Conversation:
        """Convert the input sample to a Conversation object.

        Args:
            sample: The input example

        Returns:
            Conversation: The resulting conversation
        """
        # Sample is already in conversation format
        sample_dict = sample.to_dict()
        return Conversation.from_dict(sample_dict)
