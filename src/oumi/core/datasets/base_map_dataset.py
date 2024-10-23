import gc
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, cast

import datasets
import pandas as pd
from torch.utils.data import MapDataPipe

from oumi.utils.hf_datasets_utils import is_cached_to_disk_hf_dataset
from oumi.utils.logging import logger


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

    def as_generator(self):
        """Returns a generator for the dataset."""
        for idx in range(len(self)):
            yield self[idx]

    def to_hf(self) -> datasets.Dataset:
        """Converts the dataset to a Hugging Face dataset."""
        return cast(
            datasets.Dataset, datasets.Dataset.from_generator(self.as_generator)
        )

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
