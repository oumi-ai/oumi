from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd
from typing_extensions import override

from oumi.core.datasets import BaseClassificationDataset
from oumi.core.registry import register_dataset
from oumi.core.types.classification import Classification
from oumi.utils.io_utils import load_json, load_jsonlines


@register_dataset("text_classification_jsonl")
class TextClassificationJsonLinesDataset(BaseClassificationDataset):
    """TextSftJsonLinesDataset for loading SFT data in oumi and alpaca formats.

    This dataset class is designed to work with JSON Lines (.jsonl) or
    JSON (.json) files containing text-based supervised fine-tuning (SFT) data.
    It supports loading data either from a file or from a provided list of data
    samples in oumi and alpaca formats.

    Supported formats:
    1. JSONL or JSON of conversations (Oumi format)
    2. JSONL or JSON of Alpaca-style turns (instruction, input, output)

    Args:
        dataset_path (Optional[Union[str, Path]]): Path to the dataset file
            (.jsonl or .json).
        data (Optional[List[Dict[str, Any]]]): List of conversation dicts if not
            loading from a file.
        **kwargs: Additional arguments to pass to the parent class.

    Examples:
        Loading conversations from a JSONL file with auto-detection:
            >>> dataset = TextSftJsonLinesDataset(
            ...     dataset_path="/path/to/your/dataset.jsonl"
            ... )

        Loading Alpaca-style data from a JSON file:
            >>> dataset = TextSftJsonLinesDataset(
            ...     dataset_path="/path/to/your/dataset.json",
            ...     format="alpaca"
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
        data: Optional[list[dict[str, Any]]] = None,
        **kwargs,
    ):
        """Initializes a new instance of the TextSftJsonLinesDataset class.

        Args:
            dataset_path (Optional): Path to the JSON lines dataset file.
            data (Optional): List of conversation dicts if not loading from a file.
            format (Optional): The format of the data. Either "conversations",
                or "alpaca". If not provided, the format will be
                auto-detected.
            **kwargs: Additional arguments to pass to the parent class.

        Raises:
            ValueError: If neither dataset_path nor data is provided,
                or if both are provided.
        """
        if dataset_path is not None and data is not None:
            raise ValueError(
                "Either dataset_path or data must be provided, but not both"
            )

        self._data_column: str = "_data_column"
        self._dataset_path: Optional[Path] = (
            Path(dataset_path) if dataset_path else None
        )

        if data is not None:
            data_frame = pd.DataFrame({self._data_column: data})

        elif self._dataset_path is not None:
            if self._dataset_path.suffix.lower() == ".jsonl":
                data = load_jsonlines(self._dataset_path)

            elif self._dataset_path.suffix.lower() == ".json":
                data = load_json(self._dataset_path)

            else:
                raise ValueError(
                    f"Unsupported file format: {self._dataset_path.suffix}. "
                    "Use .jsonl or .json file extensions."
                )

            data_frame = pd.DataFrame({self._data_column: data})

        else:
            raise ValueError("Dataset path or data must be provided")

        assert data_frame is not None
        self._data: pd.DataFrame = data_frame

        super().__init__(**kwargs)

    @override
    def transform_classification(self, example: dict) -> Classification:
        """Preprocesses the inputs of the example and returns a dictionary.

        Args:
            example (dict): The example containing the input and label.

        Returns:
            dict: The preprocessed inputs as a dictionary.

        """
        try:
            return Classification.model_validate(example[self._data_column])
        except Exception as e:
            raise ValueError(
                f"Invalid conversation format. "
                f"Expected a dictionary with a 'messages' key "
                f"containing a list of message dictionaries. Error: {str(e)}"
            ) from e

    @override
    def _load_data(self) -> pd.DataFrame:
        # Data is already loaded in __init__
        return self._data
