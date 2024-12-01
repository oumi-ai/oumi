import gc
import math
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, cast

import datasets
import pandas as pd
from torch.utils.data import MapDataPipe

from oumi.utils.hf_datasets_utils import is_cached_to_disk_hf_dataset
from oumi.utils.logging import logger
from oumi.utils.torch_utils import (
    estimate_sample_dict_size_in_bytes,
)


class BaseMapDataset(MapDataPipe, ABC):
    """Abstract base class for map datasets."""

    _data: pd.DataFrame
    dataset_name: str
    dataset_path: Optional[str] = None
    default_dataset: Optional[str] = None
    default_subset: Optional[str] = None
    trust_remote_code: bool

    def __init__(
        self,
        *,
        dataset_name: Optional[str],
        dataset_path: Optional[str] = None,
        subset: Optional[str] = None,
        split: Optional[str] = None,
        trust_remote_code: bool = False,
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

    #
    # Main API
    #
    def _estimate_max_element_size_bytes(self) -> int:
        """Returns an estimate of max element size in bytes."""
        total_examples = len(self)
        if total_examples <= 0:
            return 0
        sample_elements: list[dict[str, Any]] = [self[0]]
        if total_examples > 1:
            sample_elements.append(self[total_examples - 1])
        return max(
            [estimate_sample_dict_size_in_bytes(elem) for elem in sample_elements]
        )

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

    def _transform_for_hf_dataset(self, sample: pd.Series, idx: int) -> dict:
        logger.info(f"self: {type(self)}")
        logger.info(f"sample: {type(sample)}")
        logger.info(f"idx: {type(idx)}")

        logger.info(f"idx: {idx}")
        # logger.info(f"sample: {sample}")
        processed = self[idx]
        processed["pixel_values"] = processed["pixel_values"].numpy()
        logger.info(f"processed: {processed.keys()}")
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

    def as_generator(self):
        """Returns a generator for the dataset."""
        for idx in range(len(self)):
            yield self[idx]

    def as_sharded_generator(self, shards: list[tuple[int, int]]):
        """Returns a generator for the dataset."""
        for shard in shards:
            for idx in range(shard[0], shard[1]):
                yield self[idx]

    def to_hf(self) -> datasets.Dataset:
        """Converts the dataset to a Hugging Face dataset."""
        _MAX_SHARD_SIZE = 1 * 1024 * 1024 * 1024  # ~1GB

        total_examples = len(self)

        elem_size = self._estimate_max_element_size_bytes()
        elements_per_shard: int = (
            min(total_examples, _MAX_SHARD_SIZE // elem_size)
            if elem_size > 0
            else total_examples
        )
        writer_batch_size = max(min(1000, elements_per_shard), 1)

        num_shards = int(math.ceil(float(total_examples) / elements_per_shard))
        logger.info(
            f"num_shards={num_shards} examples={total_examples} elem_size={elem_size} "
            f"writer_batch_size={writer_batch_size} "
        )

        if num_shards > 1 and False:
            num_proc = 8
            num_sub_datasets = int(math.ceil(float(num_shards) / num_proc))
            sub_datasets: list[datasets.Dataset] = []
            num_examples_per_sub_dataset = int(
                math.ceil(float(total_examples) / num_sub_datasets)
            )
            for i in range(num_sub_datasets):
                sub_dataset_begin_index = i * num_examples_per_sub_dataset
                sub_dataset_end_index = min(
                    total_examples, (i + 1) * num_examples_per_sub_dataset
                )
                sub_dataset_len = sub_dataset_end_index - sub_dataset_begin_index

                starts: list[int] = list(
                    range(
                        sub_dataset_begin_index,
                        sub_dataset_end_index,
                        min(
                            max(1, elements_per_shard // 4),
                            num_examples_per_sub_dataset,
                        ),
                    )
                )
                stops: list[int] = starts[1:] + [sub_dataset_end_index]
                shards: list[tuple[int, int]] = list(zip(starts, stops))

                _START_TIME = time.perf_counter()
                logger.info("Starting generation...")
                sub_dataset = cast(
                    datasets.Dataset,
                    datasets.Dataset.from_generator(
                        self.as_sharded_generator,
                        gen_kwargs={"shards": shards},
                        # keep_in_memory=True,
                        num_proc=num_proc,
                    ),
                )

                duration_sec = time.perf_counter() - _START_TIME
                logger.info(
                    f"Finished generation of subset {i+1} of {num_sub_datasets} ! "
                    f"(num_proc={num_proc}) Duration: {duration_sec} sec. "
                    f"Speed: {sub_dataset_len/duration_sec} examples/s "
                    f"Columns: {sub_dataset.column_names} "
                    f"Cache: {sub_dataset.cache_files} "
                    f"shards: {shards} "
                )
                sub_datasets.append(sub_dataset)

            logger.info(f"Concatenating {len(sub_datasets)} datasets...")
            result = datasets.concatenate_datasets(sub_datasets)
            assert isinstance(result, datasets.Dataset)
            logger.info(
                f"Concatenated {len(sub_datasets)} datasets with {len(result)} samples!"
                f"Columns: {result.column_names} "
                f"Cache: {result.cache_files} "
                f"Result: {result}"
            )
            assert len(result) == total_examples

            logger.info("Flattenting  indices...")
            # result = result.flatten_indices(num_proc=num_proc)
            logger.info(
                "Flattened!"
                f"Columns: {result.column_names} "
                f"Cache: {result.cache_files} "
                f"Result: {result}"
            )
            assert len(result) == total_examples
            # datasets.Dataset.from_parquet()
        elif num_shards > 1 and False:
            result = datasets.Dataset.from_pandas(self._data)
            old_columns = list(result.column_names)
            result = result.map(
                self._transform_for_hf_dataset,
                with_indices=True,
                keep_in_memory=False,
                num_proc=4,
                writer_batch_size=32,
                remove_columns=old_columns,
                features=datasets.Features(
                    {
                        "input_ids": datasets.Sequence(
                            feature=datasets.Value(dtype="int32"), length=-1
                        ),
                        "attention_mask": datasets.Sequence(
                            feature=datasets.Value(dtype="int8"), length=-1
                        ),
                        "aspect_ratio_ids": datasets.Sequence(
                            feature=datasets.Value(dtype="int64"), length=-1
                        ),
                        "aspect_ratio_mask": datasets.Sequence(
                            feature=datasets.Sequence(
                                feature=datasets.Value(dtype="int64"), length=-1
                            ),
                            length=-1,
                        ),
                        "cross_attention_mask": datasets.Sequence(
                            feature=datasets.Sequence(
                                feature=datasets.Sequence(
                                    feature=datasets.Value(dtype="int64"), length=-1
                                ),
                                length=-1,
                            ),
                            length=-1,
                        ),
                        "labels": datasets.Sequence(
                            feature=datasets.Value(dtype="int64"), length=-1
                        ),
                        "pixel_values": datasets.Array5D(
                            dtype="float32", shape=(1, 4, 3, 560, 560)
                        ),
                    }
                ),
            )
            # result.save_to_disk()
            # result.set_format(type="torch", columns=["pixel_values"])
            logger.info(
                "Flattened!"
                f"Columns: {result.column_names} "
                f"Cache: {result.cache_files} "
                f"Result: {result}"
            )
        elif num_shards > 0:
            starts: list[int] = list(
                range(
                    0,
                    total_examples,
                    writer_batch_size,
                )
            )
            stops: list[int] = starts[1:] + [total_examples]
            shards: list[tuple[int, int]] = list(zip(starts, stops))

            result = cast(
                datasets.Dataset,
                datasets.Dataset.from_generator(
                    self.as_sharded_generator,
                    gen_kwargs={"shards": shards},
                    # keep_in_memory=True,
                    num_proc=4,
                    features=datasets.Features(
                        {
                            "input_ids": datasets.Sequence(
                                feature=datasets.Value(dtype="int32"), length=-1
                            ),
                            "attention_mask": datasets.Sequence(
                                feature=datasets.Value(dtype="int8"), length=-1
                            ),
                            "aspect_ratio_ids": datasets.Sequence(
                                feature=datasets.Value(dtype="int64"), length=-1
                            ),
                            "aspect_ratio_mask": datasets.Sequence(
                                feature=datasets.Sequence(
                                    feature=datasets.Value(dtype="int64"), length=-1
                                ),
                                length=-1,
                            ),
                            "cross_attention_mask": datasets.Sequence(
                                feature=datasets.Sequence(
                                    feature=datasets.Sequence(
                                        feature=datasets.Value(dtype="int64"), length=-1
                                    ),
                                    length=-1,
                                ),
                                length=-1,
                            ),
                            "labels": datasets.Sequence(
                                feature=datasets.Value(dtype="int64"), length=-1
                            ),
                            "pixel_values": datasets.Array5D(
                                dtype="float32", shape=(1, 4, 3, 560, 560)
                            ),
                        }
                    ),
                    writer_batch_size=writer_batch_size,
                ),
            )
        else:
            result = cast(
                datasets.Dataset,
                datasets.Dataset.from_generator(
                    self.as_generator,
                    features=datasets.Features(
                        {
                            "input_ids": datasets.Sequence(
                                feature=datasets.Value(dtype="int32"), length=-1
                            ),
                            "attention_mask": datasets.Sequence(
                                feature=datasets.Value(dtype="int8"), length=-1
                            ),
                            "aspect_ratio_ids": datasets.Sequence(
                                feature=datasets.Value(dtype="int64"), length=-1
                            ),
                            "aspect_ratio_mask": datasets.Sequence(
                                feature=datasets.Sequence(
                                    feature=datasets.Value(dtype="int64"), length=-1
                                ),
                                length=-1,
                            ),
                            "cross_attention_mask": datasets.Sequence(
                                feature=datasets.Sequence(
                                    feature=datasets.Sequence(
                                        feature=datasets.Value(dtype="int64"), length=-1
                                    ),
                                    length=-1,
                                ),
                                length=-1,
                            ),
                            "labels": datasets.Sequence(
                                feature=datasets.Value(dtype="int64"), length=-1
                            ),
                            "pixel_values": datasets.Array5D(
                                dtype="float32", shape=(1, 4, 3, 560, 560)
                            ),
                        }
                    ),
                    writer_batch_size=writer_batch_size,
                ),
            )
            logger.info(f"Features: {result.features}")
            logger.info(f"Dataset: {result}")
            logger.info(f"Arrow schema: {result.features.arrow_schema}")

        # result.map(
        #     lambda sample: self.__getitem__(int(sample["index"])), num_proc=num_proc
        # )
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
            # num_proc=4,
            # download_mode=datasets.DownloadMode.REUSE_CACHE_IF_EXISTS,
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
