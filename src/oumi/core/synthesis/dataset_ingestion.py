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

from enum import Enum
from pathlib import Path
from typing import Optional

import pandas as pd
from datasets import load_dataset

from oumi.core.configs.params.synthesis_params import DatasetSource


class DatasetStorageType(Enum):
    """Storage location for a dataset (local, HuggingFace, etc.)."""

    HF = "hf"
    """HuggingFace"""

    LOCAL = "local"
    """Local files"""


class DatasetPath:
    """Path to a dataset in some storage location."""

    def __init__(self, path: str):
        """Initialize the dataset path.

        Args:
            path: The path to the dataset.

        Possible path formats:
        HuggingFace:
        - "hf:repo_id/dataset_name"
        Local:
        - "path/to/data/file.jsonl"
        - "path/to/data/file.csv"
        - "path/to/data/file.tsv"
        - "path/to/data/file.parquet"
        - "path/to/data/file.json"
        - "path/to/data/*.jsonl"
        - "path/to/data/*.csv"
        - "path/to/data/*.tsv"
        - "path/to/data/*.parquet"
        - "path/to/data/*.json"
        """
        self._path = path
        self._storage_type = self._get_storage_type(path)
        self._file_extension = ""
        if self._storage_type == DatasetStorageType.LOCAL:
            self._file_extension = self._get_file_extension(path)
            if self._file_extension not in ["jsonl", "csv", "tsv", "parquet", "json"]:
                raise ValueError(f"Invalid path: {path}")

    def _get_storage_type(self, path: str) -> DatasetStorageType:
        """Get the storage type from the path."""
        prefix = path.split(":")[0]
        if prefix == "hf":
            return DatasetStorageType.HF
        else:
            return DatasetStorageType.LOCAL

    def _get_file_extension(self, path: str) -> str:
        """Get the file extension from the path."""
        return Path(path).suffix.lower()[1:]

    def get_path_str(self) -> str:
        """Get the path."""
        if self._storage_type == DatasetStorageType.HF:
            return self._path.split(":")[1]
        else:
            return self._path

    def get_storage_type(self) -> DatasetStorageType:
        """Get the storage type."""
        return self._storage_type

    def get_file_extension(self) -> str:
        """Get the file extension."""
        return self._file_extension


class DatasetReader:
    """Reads a dataset from some storage location.

    Supports:
    - HuggingFace
    - Local files (JSONL, CSV, TSV, Parquet, JSON)
    - Glob patterns
    """

    def read(self, data_source: DatasetSource) -> list[dict]:
        """Read the data from the data path."""
        data_path = DatasetPath(data_source.path)
        storage_type = data_path.get_storage_type()
        file_extension = data_path.get_file_extension()
        if storage_type == DatasetStorageType.HF:
            samples = self._read_from_hf(
                data_path.get_path_str(),
                data_source.hf_split,
                data_source.hf_revision,
            )
        elif storage_type == DatasetStorageType.LOCAL:
            if "*" in data_path.get_path_str():
                samples = self._read_from_glob(data_path.get_path_str(), file_extension)
            else:
                samples = self._read_from_local(
                    data_path.get_path_str(),
                    file_extension,
                )
        else:
            raise ValueError(f"Unsupported storage type: {storage_type}")

        if data_source.attribute_map:
            samples = self._map_attributes(samples, data_source.attribute_map)

        return samples

    def _map_attributes(
        self,
        samples: list[dict],
        attribute_map: dict[str, str],
    ) -> list[dict]:
        """Map the attributes to the samples."""
        new_samples = []
        for sample in samples:
            new_sample = {}
            for old_key, new_key in attribute_map.items():
                new_sample[new_key] = sample[old_key]
            new_samples.append(new_sample)
        return new_samples

    def _read_from_hf(
        self,
        hf_path: str,
        split: Optional[str] = None,
        revision: Optional[str] = None,
    ) -> list[dict]:
        """Read the dataset from HuggingFace."""
        dataset = load_dataset(hf_path, split=split, revision=revision)
        samples = []
        for sample in dataset:
            if not isinstance(sample, dict):
                raise ValueError(f"Expected dict, got {type(sample)}")
            samples.append(sample)
        return samples

    def _read_from_local(self, local_path: str, file_extension: str) -> list[dict]:
        """Read the dataset from the local path."""
        if file_extension == "jsonl":
            return self._read_from_jsonl(local_path)
        elif file_extension == "csv":
            return self._read_from_csv(local_path, sep=",")
        elif file_extension == "tsv":
            return self._read_from_csv(local_path, sep="\t")
        elif file_extension == "parquet":
            return self._read_from_parquet(local_path)
        elif file_extension == "json":
            return self._read_from_json(local_path)
        else:
            raise ValueError(f"Unsupported local path suffix: {file_extension}")

    def _read_from_glob(self, glob_path: str, file_extension: str) -> list[dict]:
        """Read the data from the glob path."""
        path = Path(glob_path)
        files = Path(path.parent).glob(path.name)
        samples = []
        for file in files:
            samples.extend(self._read_from_local(file.as_posix(), file_extension))
        return samples

    def _read_from_jsonl(self, jsonl_path: str) -> list[dict]:
        """Read the dataset from a JSONL file."""
        jsonl_df = pd.read_json(jsonl_path, lines=True)
        return jsonl_df.to_dict(orient="records")

    def _read_from_csv(self, csv_path: str, sep: str = ",") -> list[dict]:
        """Read the dataset from a CSV file."""
        csv_df = pd.read_csv(csv_path, sep=sep)
        return csv_df.to_dict(orient="records")

    def _read_from_parquet(self, parquet_path: str) -> list[dict]:
        """Read the dataset from a Parquet file."""
        parquet_df = pd.read_parquet(parquet_path)
        return parquet_df.to_dict(orient="records")

    def _read_from_json(self, json_path: str) -> list[dict]:
        """Read the dataset from a JSON file."""
        json_df = pd.read_json(json_path)
        return json_df.to_dict(orient="records")
