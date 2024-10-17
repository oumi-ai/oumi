from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from typing_extensions import override

from oumi.core.datasets import BaseLMSftDataset
from oumi.core.registry import register_dataset
from oumi.core.types.conversation import Conversation
from oumi.utils.io_utils import load_jsonlines


@register_dataset("text_sft_jsonl")
class TextSftJsonLinesDataset(BaseLMSftDataset):
    """TextSftJsonLinesDataset for loading SFT data in Oumi format.

    This dataset class is designed to work with JSON Lines (.jsonl) files containing
    text-based supervised fine-tuning (SFT) data. It supports loading data either
    from a file or from a provided list of data samples.

    Usage example:
        Examples:
            Loading from a file:
                >>> dataset = TextSftJsonLinesDataset(
                ...     dataset_path="/path/to/your/dataset.jsonl",
                ...     data_column="messages"
                ... )

            Loading from a list of data samples:
                >>> data_samples = [
                ...     {"messages": [{"role": "user", "content": "Hello"},
                ...                   {"role": "assistant", "content": "Hi there!"}]},
                ...     {"messages": [{"role": "user", "content": "How are you?"},
                ...                   {"role": "assistant", "content": "great!"}]}
                ... ]
                >>> dataset = TextSftJsonLinesDataset(
                ...     data=data_samples,
                ... )
    """

    default_dataset = "custom"

    def __init__(
        self,
        dataset_path: Optional[Union[str, Path]] = None,
        data: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ):
        """Initializes a new instance of the SftJsonLinesDataset class.

        Args:
            dataset_path (Optional): Path to the JSON lines dataset file.
            data (Optional): List of conversation dicts if not loading from a file.
            **kwargs: Additional arguments to pass to the parent class.

        Raises:
            ValueError: If neither dataset_path nor data is provided,
                or if both are provided.
        """
        if dataset_path is not None and data is not None:
            raise ValueError(
                "Either dataset_path or data must be provided, but not both"
            )

        self._data_column: str = "_messages_column"
        self._dataset_path: Optional[Path] = (
            Path(dataset_path) if dataset_path else None
        )

        if data is not None:
            data_frame = pd.DataFrame({self._data_column: data})
        elif self._dataset_path is not None:
            data = load_jsonlines(self._dataset_path)
            data_frame = pd.DataFrame({self._data_column: data})
        else:
            raise ValueError("Dataset path or data must be provided")

        assert data_frame is not None
        self._data: pd.DataFrame = data_frame

        super().__init__(**kwargs)

    @override
    def _load_data(self) -> pd.DataFrame:
        # Data is already loaded in __init__
        return self._data

    @override
    def transform_conversation(self, example: dict) -> Conversation:
        """Transform a single conversation example into a Conversation object.

        Args:
            example: The input example containing the messages.

        Returns:
            Conversation: A Conversation object containing the messages.
        """
        conversation_dict = example[self._data_column]
        return Conversation.model_validate(conversation_dict)
