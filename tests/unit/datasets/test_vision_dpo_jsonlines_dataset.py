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

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import jsonlines
import pandas as pd
import pytest

from oumi.core.tokenizers.base_tokenizer import BaseTokenizer
from oumi.datasets.vision_language.vision_dpo_jsonlines import VisionDpoJsonlinesDataset

_IMAGE_TOKEN = "<image_token>"
_IMAGE_TOKEN_ID = 32001


@pytest.fixture
def mock_processor():
    processor = Mock()
    processor.processor_name = "llava-hf/llava-1.5-7b-hf"
    processor.tokenizer = Mock()
    processor.image_processor = Mock()
    processor.image_processor.size = {"longest_edge": 224}
    processor.chat_template = None
    processor.image_token = _IMAGE_TOKEN
    processor.image_token_id = _IMAGE_TOKEN_ID
    processor.label_ignore_index = -100
    return processor


@pytest.fixture
def mock_tokenizer() -> MagicMock:
    def _convert_tokens_to_ids(token: str) -> int:
        if token == _IMAGE_TOKEN:
            return _IMAGE_TOKEN_ID
        return 101

    mock = MagicMock(spec=BaseTokenizer)
    mock.pad_token_id = 0
    mock.convert_tokens_to_ids = MagicMock(side_effect=_convert_tokens_to_ids)
    return mock


@pytest.fixture
def sample_vision_dpo_data():
    return [
        {
            "prompt": "What do you see in this image?",
            "images": ["https://picsum.photos/200/300"],
            "chosen": [
                {"role": "assistant", "content": "I see a beautiful landscape."}
            ],
            "rejected": [{"role": "assistant", "content": "I see nothing."}],
        },
        {
            "prompt": "Describe the main subject of this image.",
            "images": ["path/to/local/image.jpg"],
            "chosen": [
                {
                    "role": "assistant",
                    "content": "The main subject is a golden retriever dog.",
                }
            ],
            "rejected": [
                {"role": "assistant", "content": "The main subject is a cat."}
            ],
        },
        {
            "prompt": "What activity is happening in this image?",
            "images": [],
            "chosen": [
                {
                    "role": "assistant",
                    "content": "The image shows children playing soccer.",
                }
            ],
            "rejected": [
                {"role": "assistant", "content": "The image shows people swimming."}
            ],
        },
    ]


def test_init_with_data(sample_vision_dpo_data, mock_tokenizer, mock_processor):
    dataset = VisionDpoJsonlinesDataset(
        data=sample_vision_dpo_data, tokenizer=mock_tokenizer, processor=mock_processor
    )

    assert len(dataset._data) == 3
    assert isinstance(dataset._data, pd.DataFrame)
    assert list(dataset._data.columns) == ["prompt", "images", "chosen", "rejected"]

    # Check first row
    first_row = dataset._data.iloc[0]
    assert first_row["prompt"] == "What do you see in this image?"
    assert first_row["images"] == ["https://picsum.photos/200/300"]
    assert first_row["chosen"] == [
        {"role": "assistant", "content": "I see a beautiful landscape."}
    ]
    assert first_row["rejected"] == [{"role": "assistant", "content": "I see nothing."}]


def test_init_with_dataset_path(sample_vision_dpo_data, mock_tokenizer, mock_processor):
    with tempfile.TemporaryDirectory() as folder:
        jsonl_file = Path(folder) / "vision_dpo_test.jsonl"
        with jsonlines.open(jsonl_file, mode="w") as writer:
            writer.write_all(sample_vision_dpo_data)

        dataset = VisionDpoJsonlinesDataset(
            dataset_path=str(jsonl_file),
            tokenizer=mock_tokenizer,
            processor=mock_processor,
        )

        assert len(dataset._data) == 3
        assert isinstance(dataset._data, pd.DataFrame)

        # Check that data was loaded correctly
        first_row = dataset._data.iloc[0]
        assert first_row["prompt"] == "What do you see in this image?"
        assert first_row["images"] == ["https://picsum.photos/200/300"]


def test_init_validation_errors(sample_vision_dpo_data, mock_tokenizer, mock_processor):
    # Test missing both dataset_path and data
    with pytest.raises(
        ValueError, match="Either dataset_path or data must be provided"
    ):
        VisionDpoJsonlinesDataset(tokenizer=mock_tokenizer, processor=mock_processor)

    # Test providing both dataset_path and data
    with tempfile.TemporaryDirectory() as folder:
        jsonl_file = Path(folder) / "test.jsonl"
        jsonl_file.write_text('{"test": "data"}')

        with pytest.raises(
            ValueError, match="Only one of dataset_path or data must be provided"
        ):
            VisionDpoJsonlinesDataset(
                dataset_path=str(jsonl_file),
                data=sample_vision_dpo_data,
                tokenizer=mock_tokenizer,
                processor=mock_processor,
            )


def test_transform_preference(sample_vision_dpo_data, mock_tokenizer, mock_processor):
    dataset = VisionDpoJsonlinesDataset(
        data=sample_vision_dpo_data, tokenizer=mock_tokenizer, processor=mock_processor
    )

    sample = sample_vision_dpo_data[0]

    with (
        patch.object(dataset, "_load_image") as mock_load_image,
        patch.object(dataset, "_resize_image") as mock_resize_image,
    ):
        mock_image = Mock()
        mock_load_image.return_value = mock_image
        mock_resize_image.return_value = mock_image

        result = dataset.transform_preference(sample)

        assert "prompt" in result
        assert "chosen" in result
        assert "rejected" in result
        assert "images" in result
