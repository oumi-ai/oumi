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

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from oumi.core.configs.params.synthesis_params import DatasetSource
from oumi.core.synthesis.dataset_ingestion import (
    DatasetPath,
    DatasetReader,
    DatasetStorageType,
)


def test_invalid_path():
    """Test invalid path."""
    with pytest.raises(ValueError, match="Invalid path"):
        DatasetPath("invalid:path")


def test_local_path():
    """Test initialization with local JSONL path."""
    path = DatasetPath("path/to/data/file.jsonl")
    assert path.get_path_str() == "path/to/data/file.jsonl"
    assert path.get_storage_type() == DatasetStorageType.LOCAL

    path = DatasetPath("path/to/data/file.csv")
    assert path.get_path_str() == "path/to/data/file.csv"
    assert path.get_storage_type() == DatasetStorageType.LOCAL

    path = DatasetPath("path/to/data/file.tsv")
    assert path.get_path_str() == "path/to/data/file.tsv"
    assert path.get_storage_type() == DatasetStorageType.LOCAL

    path = DatasetPath("path/to/data/file.parquet")
    assert path.get_path_str() == "path/to/data/file.parquet"
    assert path.get_storage_type() == DatasetStorageType.LOCAL

    path = DatasetPath("path/to/data/file.json")
    assert path.get_path_str() == "path/to/data/file.json"
    assert path.get_storage_type() == DatasetStorageType.LOCAL

    path = DatasetPath("path/to/data/*.jsonl")
    assert path.get_path_str() == "path/to/data/*.jsonl"
    assert path.get_storage_type() == DatasetStorageType.LOCAL


def test_huggingface_path():
    """Test get_path_str for HuggingFace path."""
    path = DatasetPath("hf:repo_id/dataset_name")
    assert path.get_path_str() == "repo_id/dataset_name"
    assert path.get_storage_type() == DatasetStorageType.HF


@pytest.fixture
def reader():
    """Create a DatasetReader instance."""
    return DatasetReader()


@pytest.fixture
def sample_data():
    """Sample data for testing."""
    return [
        {"id": 1, "name": "Alice", "age": 25},
        {"id": 2, "name": "Bob", "age": 30},
        {"id": 3, "name": "Charlie", "age": 35},
    ]


@pytest.fixture
def sample_dataframe(sample_data):
    """Sample pandas DataFrame for testing."""
    return pd.DataFrame(sample_data)


def test_read_from_huggingface(reader, sample_data):
    """Test reading from HuggingFace dataset."""
    data_source = DatasetSource(path="hf:test/dataset")

    with patch("oumi.core.synthesis.dataset_ingestion.load_dataset") as mock_load:
        mock_dataset = MagicMock()
        mock_dataset.__iter__ = MagicMock(return_value=iter(sample_data))
        mock_load.return_value = mock_dataset

        result = reader.read(data_source)

        mock_load.assert_called_once_with("test/dataset", split=None, revision=None)
        assert result == sample_data


def test_read_from_hf_with_parameters(reader, sample_data):
    """Test reading from HuggingFace with split and revision parameters."""
    with patch("oumi.core.synthesis.dataset_ingestion.load_dataset") as mock_load:
        mock_dataset = MagicMock()
        mock_dataset.__iter__ = MagicMock(return_value=iter(sample_data))
        mock_load.return_value = mock_dataset

        result = reader._read_from_hf("test/dataset", split="train", revision="v1.0")

        mock_load.assert_called_once_with(
            "test/dataset",
            split="train",
            revision="v1.0",
        )
        assert result == sample_data


def test_read_from_local_jsonl(reader, sample_data, sample_dataframe):
    """Test reading from local JSONL file."""
    data_source = DatasetSource(path="data/file.jsonl")

    with patch("pandas.read_json", return_value=sample_dataframe) as mock_read:
        result = reader.read(data_source)

        mock_read.assert_called_once_with("data/file.jsonl", lines=True)
        assert result == sample_data


def test_read_from_local_csv(reader, sample_data, sample_dataframe):
    """Test reading from local CSV file."""
    data_source = DatasetSource(path="data/file.csv")

    with patch("pandas.read_csv", return_value=sample_dataframe) as mock_read:
        result = reader.read(data_source)

        mock_read.assert_called_once_with("data/file.csv", sep=",")
        assert result == sample_data


def test_read_from_local_tsv(reader, sample_data, sample_dataframe):
    """Test reading from local TSV file."""
    data_source = DatasetSource(path="data/file.tsv")

    with patch("pandas.read_csv", return_value=sample_dataframe) as mock_read:
        result = reader.read(data_source)

        mock_read.assert_called_once_with("data/file.tsv", sep="\t")
        assert result == sample_data


def test_read_from_local_parquet(reader, sample_data, sample_dataframe):
    """Test reading from local Parquet file."""
    data_source = DatasetSource(path="data/file.parquet")

    with patch("pandas.read_parquet", return_value=sample_dataframe) as mock_read:
        result = reader.read(data_source)

        mock_read.assert_called_once_with("data/file.parquet")
        assert result == sample_data


def test_read_from_local_json(reader, sample_data, sample_dataframe):
    """Test reading from local JSON file."""
    data_source = DatasetSource(path="data/file.json")

    with patch("pandas.read_json", return_value=sample_dataframe) as mock_read:
        result = reader.read(data_source)

        mock_read.assert_called_once_with("data/file.json")
        assert result == sample_data


def test_read_from_local_glob_pattern(reader, sample_data):
    """Test reading from local files using glob pattern."""
    data_source = DatasetSource(path="data/*.jsonl")

    with patch.object(reader, "_read_from_glob", return_value=sample_data) as mock_glob:
        result = reader.read(data_source)

        mock_glob.assert_called_once_with("data/*.jsonl", "jsonl")
        assert result == sample_data


def test_read_with_attribute_mapping(reader, sample_data):
    """Test reading with attribute mapping."""
    data_source = DatasetSource(
        path="hf:test/dataset", attribute_map={"id": "new_id", "name": "full_name"}
    )

    with patch("oumi.core.synthesis.dataset_ingestion.load_dataset") as mock_load:
        mock_dataset = MagicMock()
        mock_dataset.__iter__ = MagicMock(return_value=iter(sample_data))
        mock_load.return_value = mock_dataset

        result = reader.read(data_source)

        expected = [
            {"new_id": 1, "full_name": "Alice"},
            {"new_id": 2, "full_name": "Bob"},
            {"new_id": 3, "full_name": "Charlie"},
        ]
        assert result == expected


def test_read_unsupported_storage_type(reader):
    """Test reading with unsupported storage type."""
    data_source = DatasetSource(path="data/file.json")
    data_source.path = "unsupported:data/file.txt"

    # Mock the storage type detection to return an invalid type
    with patch.object(DatasetPath, "_get_storage_type") as mock_get_type:
        mock_get_type.return_value = "UNSUPPORTED"

        with pytest.raises(ValueError, match="Unsupported storage type"):
            reader.read(data_source)


def test_read_unsupported_local_file_type(reader):
    """Test reading unsupported local file type."""
    data_source = DatasetSource(path="data/file.json")
    data_source.path = "data/file.txt"

    with pytest.raises(ValueError, match="Invalid path: data/file.txt"):
        reader.read(data_source)


def test_read_from_glob_multiple_files(reader, sample_data):
    """Test reading from multiple files using glob pattern."""
    data_source = DatasetSource(path="data/*.jsonl")

    # Mock Path.glob to return multiple files
    mock_files = [
        MagicMock(as_posix=MagicMock(return_value="data/file1.jsonl")),
        MagicMock(as_posix=MagicMock(return_value="data/file2.jsonl")),
    ]
    with patch("pathlib.Path.glob", return_value=mock_files):
        with (
            patch.object(
                reader,
                "_read_from_local",
                side_effect=[
                    sample_data[:2],  # First file has 2 records
                    sample_data[2:],  # Second file has 1 record
                ],
            ) as mock_read_local
        ):
            result = reader.read(data_source)
            assert mock_read_local.call_count == 2
            mock_read_local.assert_any_call("data/file1.jsonl", "jsonl")
            mock_read_local.assert_any_call("data/file2.jsonl", "jsonl")
            assert result == sample_data


def test_map_attributes_basic(reader):
    """Test basic attribute mapping."""
    samples = [
        {"old_key": "value1", "another_key": "value2"},
        {"old_key": "value3", "another_key": "value4"},
    ]
    attribute_map = {"old_key": "new_key"}

    result = reader._map_attributes(samples, attribute_map)

    expected = [
        {"new_key": "value1"},
        {"new_key": "value3"},
    ]
    assert result == expected


def test_map_attributes_multiple_mappings(reader):
    """Test attribute mapping with multiple mappings."""
    samples = [
        {"id": 1, "name": "Alice", "age": 25},
        {"id": 2, "name": "Bob", "age": 30},
    ]
    attribute_map = {"id": "user_id", "name": "full_name", "age": "years"}

    result = reader._map_attributes(samples, attribute_map)

    expected = [
        {"user_id": 1, "full_name": "Alice", "years": 25},
        {"user_id": 2, "full_name": "Bob", "years": 30},
    ]
    assert result == expected


def test_read_empty_dataset(reader):
    """Test reading empty dataset."""
    data_source = DatasetSource(path="hf:test/empty_dataset")

    with patch("oumi.core.synthesis.dataset_ingestion.load_dataset") as mock_load:
        mock_dataset = MagicMock()
        mock_dataset.__iter__ = MagicMock(return_value=iter([]))
        mock_load.return_value = mock_dataset

        result = reader.read(data_source)

        assert result == []
