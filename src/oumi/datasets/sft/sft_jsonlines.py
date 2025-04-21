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

from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd
from typing_extensions import override

from oumi.core.datasets import BaseSftDataset
from oumi.core.registry import register_dataset
from oumi.core.types.conversation import Conversation
from oumi.utils.io_utils import load_json, load_jsonlines


@register_dataset("text_sft")
@register_dataset("text_sft_jsonl")
class TextSftJsonLinesDataset(BaseSftDataset):
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
        format (Optional[str]): The format of the data. Either "conversations" or
            "alpaca". If not provided, the format will be auto-detected.
        **kwargs: Additional arguments to pass to the parent class.

    Examples:
        Loading conversations from a JSONL file with auto-detection:
            >>> from oumi.datasets import TextSftJsonLinesDataset
            >>> dataset = TextSftJsonLinesDataset( # doctest: +SKIP
            ...     dataset_path="/path/to/your/dataset.jsonl"
            ... )

        Loading Alpaca-style data from a JSON file:
            >>> from oumi.datasets import TextSftJsonLinesDataset
            >>> dataset = TextSftJsonLinesDataset( # doctest: +SKIP
            ...     dataset_path="/path/to/your/dataset.json",
            ...     format="alpaca"
            ... )

        Loading from a list of data samples:
            >>> from oumi.datasets import TextSftJsonLinesDataset
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
        format: Optional[str] = None,
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

        self._data_column: str = "_messages_column"
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

        if format and format not in ["oumi", "alpaca"]:
            raise ValueError(
                f"Invalid format: {format}. Supported formats are 'oumi', and 'alpaca'."
            )

        self._format: str = format if format else self._detect_format(data_frame)

        super().__init__(**kwargs)

    def _detect_format(self, data_frame: pd.DataFrame) -> str:
        """Detect the format of the data based on the first item.

        Args:
            data_frame: The DataFrame containing the data.

        Returns:
            str: The detected format ("oumi", or "alpaca").

        Raises:
            ValueError: If the format cannot be detected.
        """
        first_item = data_frame[self._data_column].iloc[0]

        if not isinstance(first_item, dict):
            raise ValueError(
                "Invalid data format. Each item in the dataset should be a dictionary. "
                f"Found type: {type(first_item)}. "
                "Please check your data structure and try again."
            )

        if "messages" in first_item:
            if isinstance(first_item["messages"], list) and all(
                isinstance(m, dict) and "role" in m and "content" in m
                for m in first_item["messages"]
            ):
                return "oumi"
            return "conversations"

        elif all(key in first_item for key in ["instruction", "input", "output"]):
            return "alpaca"

        raise ValueError(
            "Unable to auto-detect format. "
            "The data structure doesn't match any supported format. "
            "Please specify the format manually or ensure your data follows "
            "one of these structures:\n"
            "1. Conversations format: "
            "{'messages': [{'role': 'user', 'content': '...'}, ...]}\n"
            "2. Alpaca format: "
            "{'instruction': '...', 'input': '...', 'output': '...'}\n"
        )

    @override
    def _load_data(self) -> pd.DataFrame:
        # Data is already loaded in __init__
        return self._data

    @override
    def transform_conversation(self, example: dict) -> Conversation:
        """Transform a single conversation example into a Conversation object.

        Args:
            example: The input example containing the messages or Alpaca-style turn.

        Returns:
            Conversation: A Conversation object containing the messages.
        """
        conversation_dict = example[self._data_column]

        if self._format == "oumi":
            try:
                return Conversation.model_validate(conversation_dict)
            except Exception as e:
                raise ValueError(
                    f"Invalid conversation format. "
                    f"Expected a dictionary with a 'messages' key "
                    f"containing a list of message dictionaries. Error: {str(e)}"
                ) from e

        elif self._format == "alpaca":
            return self._alpaca_to_conversation(conversation_dict)

        else:
            raise ValueError(f"Unsupported format: {self._format}")

    def _alpaca_to_conversation(self, turn: dict) -> Conversation:
        """Convert an Alpaca-style turn to a Conversation object.

        Args:
            turn: A dictionary containing 'instruction', 'input', and 'output' keys.

        Returns:
            Conversation: A Conversation object representing the Alpaca-style turn.

        Raises:
            ValueError: If the turn doesn't contain the required keys.
        """
        required_keys = ["instruction", "input", "output"]

        if not all(key in turn for key in required_keys):
            raise ValueError(
                f"Invalid Alpaca format. The turn must contain all of these keys: "
                "{required_keys}. "
                f"Found keys: {list(turn.keys())}"
            )

        messages = [
            {
                "role": "user",
                "content": f"{turn['instruction']}\n\n{turn['input']}".strip(),
            },
            {"role": "assistant", "content": turn["output"]},
        ]
        return Conversation(messages=messages)
