import gc
import os
from abc import ABC, abstractmethod
from collections.abc import Generator, Iterable
from pathlib import Path
from typing import Any, NamedTuple, Optional, Union, cast

import datasets
import pandas as pd
from torch.utils.data import MapDataPipe

from oumi.utils.hf_datasets_utils import is_cached_to_disk_hf_dataset
from oumi.utils.logging import logger
from oumi.utils.torch_utils import estimate_sample_dict_size_in_bytes, get_shape_as_list


class _ShardIndexRange(NamedTuple):
    start_index: int
    end_index: int


class _InferredFeatureMap(NamedTuple):
    feature_map: datasets.Features
    """Inferred feature map."""

    is_feature_map_optimized: bool
    """Indicates whether the original feature map was optimized.

    In optimized feature maps, large features use the inferred `ArrayXD` arrow
    feature type (not `sequence`) which supports datasets with more elements.
    """

    element_size_in_bytes: int
    """Estimated element size in bytes."""


class BaseMapDataset(MapDataPipe, ABC):
    """Abstract base class for map datasets."""

    _data: pd.DataFrame
    dataset_name: str
    dataset_path: Optional[str] = None
    default_dataset: Optional[str] = None
    default_subset: Optional[str] = None
    trust_remote_code: bool
    num_proc_transform: Optional[Union[str, int]] = None

    def __init__(
        self,
        *,
        dataset_name: Optional[str],
        dataset_path: Optional[str] = None,
        subset: Optional[str] = None,
        split: Optional[str] = None,
        trust_remote_code: bool = False,
        num_proc_transform: Optional[Union[str, int]] = None,
        **kwargs,
    ) -> None:
        """Initializes a new instance of the BaseDataset class."""
        dataset_type_name = self.__class__.__name__
        logger.info(f"Creating map dataset (type: {dataset_type_name})...")
        if len(kwargs) > 0:
            logger.debug(
                f"Unknown arguments: {', '.join(kwargs.keys())}. "
                "Please check the class constructor for supported arguments "
                f"(type: {dataset_type_name})."
            )

        dataset_name = dataset_name or self.default_dataset

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
        self.num_proc_transform = num_proc_transform

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

    def _as_sharded_generator(
        self, shards: list[_ShardIndexRange]
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
            datasets.Dataset.from_generator(_dummy_generator),
        )
        if len(sample_dataset) <= 0:
            raise ValueError("Empty sample dataset!")

        max_elem_bytes = max(
            [estimate_sample_dict_size_in_bytes(elem) for elem in samples_list]
        )

        features = sample_dataset.features.copy()
        is_feature_map_optimized: bool = False

        # At this time, we care mostly about `pixel_values` as it's by far the largest
        # feature (e.g., 15MB for Llama 3.2 Vision), which causes serialization errors
        # for large datasets if saved in the default format, which is
        # a nested sequence (of sequences (of sequences ...)).
        # TODO: Tune feature types for other features for efficiency.
        if "pixel_values" in samples_list[0]:
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

        return _InferredFeatureMap(
            feature_map=features,
            is_feature_map_optimized=is_feature_map_optimized,
            element_size_in_bytes=max_elem_bytes,
        )

    def to_hf(self) -> datasets.Dataset:
        """Converts the dataset to a Hugging Face dataset."""
        _MAX_SHARD_SIZE = 1 * 1024 * 1024 * 1024  # ~1GB

        num_proc = None
        if self.num_proc_transform is not None:
            if isinstance(self.num_proc_transform, int):
                num_proc = self.num_proc_transform
            elif self.num_proc_transform == "auto":
                num_proc = os.cpu_count()

        assert (
            num_proc is None or num_proc > 0
        ), f"num_proc_transform: {self.num_proc_transform}"

        num_proc = max(1, num_proc if num_proc is not None else 1)
        total_examples = len(self)
        output_features: _InferredFeatureMap = (
            self._detect_features_and_estimate_element_size_bytes(
                self._as_sharded_generator(
                    [_ShardIndexRange(start_index=0, end_index=min(5, total_examples))]
                )
            )
        )
        elements_per_shard: int = (
            min(
                total_examples, _MAX_SHARD_SIZE // output_features.element_size_in_bytes
            )
            if output_features.element_size_in_bytes
            else total_examples
        )
        writer_batch_size = max(min(1000, elements_per_shard), 1)

        logger.debug(
            f"features={output_features} examples={total_examples} "
            f"writer_batch_size={writer_batch_size} "
        )

        # If feature map isn't "optimized" then ignore it to fallback
        # to the default behavior in `from_generator()`.
        feature_map = (
            output_features.feature_map
            if output_features.is_feature_map_optimized
            else None
        )

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
            shards: list[_ShardIndexRange] = [
                _ShardIndexRange(start_index=item[0], end_index=item[1])
                for item in zip(starts, stops)
            ]

            result = cast(
                datasets.Dataset,
                datasets.Dataset.from_generator(
                    self._as_sharded_generator,
                    gen_kwargs={"shards": shards},
                    keep_in_memory=False,
                    num_proc=(num_proc if num_proc > 1 else None),
                    features=feature_map,
                    writer_batch_size=writer_batch_size,
                ),
            )
        else:
            result = cast(
                datasets.Dataset,
                datasets.Dataset.from_generator(
                    self.as_generator,
                    keep_in_memory=False,
                    features=feature_map,
                    writer_batch_size=writer_batch_size,
                ),
            )

        logger.debug(f"Dataset: {result}")
        logger.debug(f"Arrow schema: {result.features.arrow_schema}")
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
            f"Loaded DataFrame with shape: {result.shape}. Columns:\n"
            f"{result.dtypes}"
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
