from pathlib import Path
from typing import Optional, Union

import pandas as pd
from typing_extensions import override

from oumi.core.datasets import VisionLanguageSftDataset
from oumi.core.registry import register_dataset
from oumi.core.types.turn import Conversation
from oumi.utils.io_utils import load_jsonlines


@register_dataset("vision_language_jsonl")
class VLJsonlinesDataset(VisionLanguageSftDataset):
    """VLJsonlinesDataset for loading Vision-Language SFT data in Oumi format.

    This dataset class is designed to work with JSON Lines (.jsonl) files containing
    Vision-Language supervised fine-tuning (SFT) data. It supports loading data either
    from a file or from a provided list of data samples.

    Usage example:
        Examples:
            Loading from a file:
                >>> dataset = VLJsonlinesDataset(
                ...     dataset_path="/path/to/your/dataset.jsonl",
                ... )

            Loading from a list of data samples:
                >>> data_samples = [
                ...     {
                ...         "messages": [
                ...             {
                ...                 "role": "user",
                ...                 "content": "Describe this image:",
                ...                 "type": "text"
                ...             },
                ...             {
                ...                 "role": "user",
                ...                 "content": "path/to/image.jpg",
                ...                 "type": "image_path"
                ...             },
                ...             {
                ...                 "role": "assistant",
                ...                 "content": "A scenic view of the puget sound.",
                ...                 "type": "text",
                ...             },
                ...         ]
                ...     }
                ... ]
                ... ]
                >>> dataset = VLJsonlinesDataset(
                ...     data=data_samples,
                ... )
    """

    default_dataset = "custom"

    def __init__(
        self,
        dataset_path: Optional[Union[str, Path]] = None,
        data: Optional[list] = None,
        **kwargs,
    ):
        """Initializes a new instance of the VLJsonlinesDataset class."""
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
        # no-op, data is already loaded in __init__
        return self._data

    @override
    def transform_conversation(self, example: dict) -> Conversation:
        """Transform a single conversation example into a Conversation object."""
        messages = example[self._data_column]
        return Conversation.model_validate(messages)
