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

import gc
import math
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Generator, Iterable, Sized
from pathlib import Path
from typing import Any, NamedTuple, Optional, Union, cast

import datasets
import pandas as pd
from torch.utils.data import MapDataPipe

from oumi.utils.hf_utils import is_cached_to_disk_hf_dataset
from oumi.utils.logging import logger
from oumi.utils.torch_utils import estimate_sample_dict_size_in_bytes, get_shape_as_list


class _ExamplesIndicesRange(NamedTuple):
    """A valid sub-range of example indices."""

    start_index: int
    end_index: int


class _InferredFeatureMap(NamedTuple):
    feature_map: datasets.Features
    """Inferred feature map."""

    is_feature_map_optimized: bool
    """Indicates whether the original feature map was optimized.

    In optimized feature maps, large features use the inferred `ArrayXD` arrow
    column type (not `sequence`) which supports larger datasets with more elements.
    """

    element_size_in_bytes: int
    """Estimated element size in bytes."""

    multimodal: bool
    """Whether the features are multimodal."""


class BaseMapDataset(MapDataPipe, Sized, ABC):
    """Abstract base class for map datasets."""

    _data: pd.DataFrame
    dataset_name: str
    dataset_path: Optional[str] = None
    default_dataset: Optional[str] = None
    default_subset: Optional[str] = None
    trust_remote_code: bool
    transform_num_workers: Optional[Union[str, int]] = None

    def __init__(
        self,
        *,
        dataset_name: Optional[str],
        dataset_path: Optional[str] = None,
        subset: Optional[str] = None,
        split: Optional[str] = None,
        trust_remote_code: bool = False,
        transform_num_workers: Optional[Union[str, int]] = None,
        **kwargs,
    ) -> None:
        """Initializes a new instance of the BaseDataset class."""
        dataset_type_name = self.__class__.__name__
        if len(kwargs) > 0:
            logger.debug(
                f"Unknown arguments: {', '.join(kwargs.keys())}. "
                "Please check the class constructor for supported arguments "
                f"(type: {dataset_type_name})."
            )

        dataset_name = dataset_name or self.default_dataset

        logger.info(
            f"Creating map dataset (type: {dataset_type_name})..."
            + (f" dataset_name: '{dataset_name}'" if dataset_name else "")
            + (f" dataset_path: '{dataset_path}'" if dataset_path else "")
        )
        if dataset_name is None:
            raise ValueError(
                "Please specify a dataset_name or "
                "set the default_dataset class attribute "
                f"(type: {dataset_type_name})."
            )

        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.dataset_subset = subset or self.default_subset
        self.split = split
        self.trust_remote_code = trust_remote_code
        self.transform_num_workers = transform_num_workers

    #
    # Main API
    #
    def __getitem__(self, idx: int) -> dict:
        """Gets the item at the specified index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            dict: The item at the specified index.
        """
        sample = self.raw(idx)
        processed = self.transform(sample)
        return processed

    def __len__(self) -> int:
        """Gets the number of items in the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        return len(self._data)

    @property
    def data(self) -> pd.DataFrame:
        """Returns the underlying dataset data."""
        return self._data

    def raw(self, idx: int) -> pd.Series:
        """Returns the raw data at the specified index.

        Args:
            idx (int): The index of the data to retrieve.

        Returns:
            pd.Series: The raw data at the specified index.
        """
        return self._data.iloc[idx]

    def as_generator(self) -> Generator[dict[str, Any], None, None]:
        """Returns a generator for the dataset."""
        for idx in range(len(self)):
            yield self[idx]

    def _as_generator_over_shards(
        self, shards: list[_ExamplesIndicesRange]
    ) -> Generator[dict[str, Any], None, None]:
        """Returns a sharded generator for the dataset."""
        for shard in shards:
            for idx in range(shard.start_index, shard.end_index):
                yield self[idx]

    def _detect_features_and_estimate_element_size_bytes(
        self, samples_iter: Iterable[dict[str, Any]]
    ) -> _InferredFeatureMap:
        """Returns an estimate of max element size in bytes."""
        samples_list = list(samples_iter)

        def _dummy_generator():
            yield from samples_list

        sample_dataset = cast(
            datasets.Dataset,
            datasets.Dataset.from_generator(_dummy_generator, keep_in_memory=True),
        )
        if len(sample_dataset) <= 0:
            raise ValueError("Empty sample dataset!")

        max_elem_bytes = max(
            [estimate_sample_dict_size_in_bytes(elem) for elem in samples_list]
        )

        features = sample_dataset.features.copy()
        is_feature_map_optimized: bool = False

        is_multimodal: bool = False
        # At this time, we care mostly about `pixel_values` as it's by far the largest
        # feature (e.g., 15MB for Llama 3.2 Vision), which causes serialization errors
        # for large datasets if saved in the default format, which is
        # a nested sequence (of sequences (of sequences ...)).
        # TODO: Tune feature types for other features for efficiency.
        if "pixel_values" in samples_list[0]:
            is_multimodal = True
            inferred_features = []
            variable_shapes_detected: bool = False
            for elem in samples_list:
                shape = tuple(get_shape_as_list(elem["pixel_values"]))
                shape_dims = len(shape)
                if shape_dims == 2:
                    feature_def = datasets.Array2D(dtype="float32", shape=shape)
                elif shape_dims == 3:
                    feature_def = datasets.Array3D(dtype="float32", shape=shape)
                elif shape_dims == 4:
                    feature_def = datasets.Array4D(dtype="float32", shape=shape)
                elif shape_dims == 5:
                    feature_def = datasets.Array5D(dtype="float32", shape=shape)
                else:
                    raise ValueError(
                        "The `pixel_values` feature has unsupported dimensionality "
                        f"({shape_dims}D). Must be 2D...5D."
                    )
                inferred_features.append(feature_def)

            for i in range(1, len(samples_list)):
                if (
                    type(inferred_features[i - 1]),
                    inferred_features[i - 1].dtype,
                    inferred_features[i - 1].shape,
                ) != (
                    type(inferred_features[i]),
                    inferred_features[i].dtype,
                    inferred_features[i].shape,
                ):
                    variable_shapes_detected = True
                    logger.warning(
                        f"The `pixel_values` feature has variable shapes: "
                        f"{inferred_features[i - 1]} vs {inferred_features[i]}!"
                    )

            if not variable_shapes_detected:
                # Re-define the feature to be `ArrayXD`
                # if all shapes are the same.
                features["pixel_values"] = inferred_features[0]
                is_feature_map_optimized = True
                logger.info(
                    "The `pixel_values` feature has this inferred type: "
                    f"{inferred_features[0]}"
                )

        del sample_dataset
        return _InferredFeatureMap(
            feature_map=features,
            is_feature_map_optimized=is_feature_map_optimized,
            element_size_in_bytes=max_elem_bytes,
            multimodal=is_multimodal,
        )

    def _compute_effective_transform_num_workers(self) -> int:
        """Returns an effective number of dataset transform workers.

        Guaranteed to be a positive integer (>= 1). 1 if no parallelism is used.
        """
        num_proc = None
        if self.transform_num_workers is not None:
            if isinstance(self.transform_num_workers, int):
                num_proc = self.transform_num_workers
            elif self.transform_num_workers == "auto":
                num_proc = os.cpu_count()
                if num_proc is not None:
                    # Limit the max number of sub-processes.
                    num_proc = min(8, num_proc)

        assert num_proc is None or num_proc > 0, (
            f"transform_num_workers: {self.transform_num_workers}"
        )

        num_proc = max(1, num_proc if num_proc is not None else 1)
        assert num_proc >= 1
        return num_proc

    def to_hf(
        self, return_iterable: bool = False
    ) -> Union[datasets.Dataset, datasets.IterableDataset]:
        """Converts the dataset to a Hugging Face dataset.

        Args:
            return_iterable: Whether to return an iterable dataset.
                Iterable datasets aren't cached to disk, which can sometimes be
                advantageous. For example, if transformed examples are very large
                (e.g., if `pixel_values` are large for multimodal data), or if you don't
                want to post-process the whole dataset before training starts.

        Returns:
            A HuggingFace dataset. Can be `datasets.Dataset` or
            `datasets.IterableDataset` depending on the value of `return_iterable`.
        """
        _MAX_SHARD_SIZE = 1 * 1024 * 1024 * 1024  # ~1GB
        dataset_type_name = self.__class__.__name__
        num_proc = self._compute_effective_transform_num_workers()
        total_examples = len(self)
        output_features: _InferredFeatureMap = (
            self._detect_features_and_estimate_element_size_bytes(
                self._as_generator_over_shards(
                    [
                        _ExamplesIndicesRange(start_index=i, end_index=(i + 1))
                        for i in range(0, total_examples, max(1, total_examples // 8))
                    ]
                )
            )
        )
        elements_per_shard: int = int(math.ceil(float(total_examples) / num_proc))
        if output_features.element_size_in_bytes > 0:
            elements_per_shard = min(
                elements_per_shard,
                _MAX_SHARD_SIZE // output_features.element_size_in_bytes,
            )
        # Clamp `writer_batch_size` to [1, 200/1000] range.
        writer_batch_size = max(
            1, min(elements_per_shard, 200 if output_features.multimodal else 1000)
        )

        logger.info(
            f"{dataset_type_name}: features={output_features.feature_map.keys()}"
        )
        logger.debug(
            f"{dataset_type_name}: features={output_features} "
            f"examples={total_examples} "
            f"writer_batch_size={writer_batch_size} num_proc={num_proc}"
        )

        # If feature map isn't "optimized" then ignore it to fallback
        # to the default behavior in `from_generator()`.
        feature_map = (
            output_features.feature_map
            if output_features.is_feature_map_optimized
            else None
        )

        start_time = time.perf_counter()
        if num_proc > 1 or (
            output_features.element_size_in_bytes * total_examples > _MAX_SHARD_SIZE
        ):
            starts: list[int] = list(
                range(
                    0,
                    total_examples,
                    writer_batch_size,
                )
            )
            stops: list[int] = starts[1:] + [total_examples]
            shards: list[_ExamplesIndicesRange] = [
                _ExamplesIndicesRange(start_index=item[0], end_index=item[1])
                for item in zip(starts, stops)
            ]

            if return_iterable:
                result = datasets.IterableDataset.from_generator(
                    self._as_generator_over_shards,
                    gen_kwargs={"shards": shards},
                    features=feature_map,
                )
            else:
                result = datasets.Dataset.from_generator(
                    self._as_generator_over_shards,
                    gen_kwargs={"shards": shards},
                    keep_in_memory=False,
                    num_proc=(num_proc if num_proc > 1 else None),
                    features=feature_map,
                    writer_batch_size=writer_batch_size,
                )
        else:
            if return_iterable:
                result = datasets.IterableDataset.from_generator(
                    self.as_generator,
                    features=feature_map,
                )
            else:
                result = datasets.Dataset.from_generator(
                    self.as_generator,
                    keep_in_memory=False,
                    features=feature_map,
                    writer_batch_size=writer_batch_size,
                )
        duration_sec = time.perf_counter() - start_time

        logger.info(
            f"Finished transforming dataset ({dataset_type_name})! "
            f"Speed: {total_examples / duration_sec:.2f} examples/sec. "
            f"Examples: {total_examples}. "
            f"Duration: {duration_sec:.1f} sec. Transform workers: {num_proc}."
        )

        if return_iterable:
            result = cast(datasets.IterableDataset, result)
            logger.debug(f"{dataset_type_name}: IterableDataset: {result}")
        else:
            result = cast(datasets.Dataset, result)
            logger.debug(
                f"{dataset_type_name}: MapDataset: {result}\n\n"
                f"Arrow schema: {result.features.arrow_schema}"
            )
        return result

    #
    # Abstract Methods
    #
    @abstractmethod
    def transform(self, sample: pd.Series) -> dict:
        """Preprocesses the inputs in the given sample.

        Args:
            sample (dict): A dictionary containing the input data.

        Returns:
            dict: A dictionary containing the preprocessed input data.
        """
        raise NotImplementedError

    #
    # Data Loading
    #
    def _load_data(self) -> pd.DataFrame:
        """Loads the dataset from the specified source.

        Returns:
            dict: The loaded dataset.
        """
        if self.dataset_path:
            result = self._load_local_dataset(self.dataset_path)
        else:
            result = self._load_hf_hub_dataset()

        # Reclaim memory after data loading.
        gc.collect()

        logger.info(
            f"Loaded DataFrame with shape: {result.shape}. Columns:\n{result.dtypes}"
        )
        return result

    def _load_local_dataset(self, path: str) -> pd.DataFrame:
        """Loads the dataset from the specified local source.

        Returns:
            dict: The loaded dataset.
        """
        dataset_path = Path(path)

        if not dataset_path.exists():
            raise FileNotFoundError(f"File not found: {dataset_path}")

        if dataset_path.suffix.lower() == ".jsonl" and dataset_path.is_file():
            result = self._load_jsonl_dataset(dataset_path)

        elif dataset_path.suffix.lower() == ".parquet" and dataset_path.is_file():
            result = self._load_parquet_dataset(dataset_path)

        elif is_cached_to_disk_hf_dataset(dataset_path):
            result = self._load_dataset_from_disk(dataset_path)

        else:
            raise ValueError(f"File format not supported for {self.dataset_name}")

        return result

    def _load_hf_hub_dataset(self) -> pd.DataFrame:
        """Loads the dataset from the specified Hugging Face Hub source.

        Returns:
            dict: The loaded dataset.
        """
        splits_or_dataset = datasets.load_dataset(
            path=self.dataset_name,
            name=self.dataset_subset,
            split=self.split,
            trust_remote_code=self.trust_remote_code,
        )

        if isinstance(
            splits_or_dataset, (datasets.IterableDataset, datasets.IterableDatasetDict)
        ):
            raise ValueError("IterableDataset is not supported with this class.")

        # Grab a single dataset split
        if isinstance(splits_or_dataset, datasets.Dataset):
            dataset = splits_or_dataset
        elif self.split is not None:
            dataset = splits_or_dataset[self.split]
        elif len(splits_or_dataset) == 1:
            dataset = splits_or_dataset.values().__iter__().__next__()
        else:
            raise ValueError(
                "Multiple splits found in the dataset. Please specify a single split. "
                f"Available splits: {list(splits_or_dataset.keys())}"
            )

        logger.info(
            "\n".join(
                [
                    "Dataset Info:",
                    f"\tSplit: {dataset.split}",
                    f"\tVersion: {dataset.version}",
                    f"\tDataset size: {dataset.dataset_size}",
                    f"\tDownload size: {dataset.download_size}",
                    f"\tSize: {dataset.size_in_bytes} bytes",
                    f"\tRows: {len(dataset)}",
                    f"\tColumns: {dataset.column_names}",
                ]
            )
        )

        result = dataset.to_pandas()
        del dataset
        return cast(pd.DataFrame, result)

    def _load_jsonl_dataset(self, path: Path) -> pd.DataFrame:
        return pd.read_json(path, lines=True)

    def _load_parquet_dataset(self, path: Path) -> pd.DataFrame:
        return pd.read_parquet(path)

    def _load_dataset_from_disk(self, path: Path) -> pd.DataFrame:
        dataset: datasets.Dataset = datasets.Dataset.load_from_disk(path)
        result = dataset.to_pandas()
        del dataset
        return cast(pd.DataFrame, result)
