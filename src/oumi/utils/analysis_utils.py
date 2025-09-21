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

import logging
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from oumi.builders.models import build_tokenizer
from oumi.core.configs.analyze_config import AnalyzeConfig
from oumi.core.configs.params.model_params import ModelParams
from oumi.core.datasets.base_map_dataset import BaseMapDataset
from oumi.core.registry.registry import REGISTRY

logger = logging.getLogger(__name__)


def build_tokenizer_from_config(tokenizer_config: Optional[dict[str, Any]]):
    """Build a tokenizer from configuration dictionary.

    Args:
        tokenizer_config: Dictionary containing tokenizer configuration

    Returns:
        Built tokenizer or None if config is None

    Raises:
        ValueError: If required fields are missing from tokenizer_config
    """
    if not tokenizer_config:
        return None

    if "model_name" not in tokenizer_config:
        raise ValueError("tokenizer_config must contain 'model_name' field")

    model_params = ModelParams(
        model_name=tokenizer_config["model_name"],
        tokenizer_kwargs=tokenizer_config.get("tokenizer_kwargs", {}),
        trust_remote_code=tokenizer_config.get("trust_remote_code", False),
    )
    tokenizer = build_tokenizer(model_params)
    logger.info(f"Built tokenizer for model: {model_params.model_name}")
    return tokenizer


def load_dataset_from_config(
    config: AnalyzeConfig, tokenizer: Optional[Any] = None
) -> BaseMapDataset:
    """Load dataset based on configuration.

    This function loads datasets directly from the registry for analysis purposes.
    If a tokenizer is provided, it will be passed to the dataset constructor.

    For custom datasets, it supports loading from local files using
    TextSftJsonLinesDataset for text data and VLJsonlinesDataset for
    vision-language data.

    Args:
        config: Configuration object containing dataset parameters
        tokenizer: Optional tokenizer to use with the dataset

    Returns:
        Loaded dataset
    """
    dataset_name = config.dataset_name
    split = config.split
    subset = config.subset
    dataset_path = config.dataset_path
    dataset_format = config.dataset_format

    if not dataset_name and not dataset_path:
        raise ValueError("Either dataset_name or dataset_path must be provided")

    # Handle custom dataset loading from local files
    if dataset_path:
        return _load_custom_dataset_from_path(
            dataset_path, dataset_format, tokenizer, config
        )

    # Handle registered dataset loading
    try:
        # Load dataset from the REGISTRY
        if dataset_name is None:
            raise ValueError("dataset_name cannot be None for registered datasets")
        dataset_class = REGISTRY.get_dataset(dataset_name, subset=subset)

        if dataset_class is not None:
            # Prepare dataset constructor arguments
            dataset_kwargs = {
                "dataset_name": dataset_name,
                "dataset_path": None,
                "split": split,
                "subset": subset,
                "trust_remote_code": config.trust_remote_code,
            }

            # Add tokenizer if provided
            if tokenizer is not None:
                dataset_kwargs["tokenizer"] = tokenizer

            # Add processor parameters for vision-language datasets
            if config.processor_name:
                dataset_kwargs["processor_name"] = config.processor_name
                dataset_kwargs["processor_kwargs"] = config.processor_kwargs
                dataset_kwargs["trust_remote_code"] = config.trust_remote_code

            # Load registered dataset with parameters
            dataset = dataset_class(**dataset_kwargs)

            # Ensure we return a BaseMapDataset
            if isinstance(dataset, BaseMapDataset):
                return dataset
            else:
                raise NotImplementedError(
                    f"Dataset type {type(dataset)} is not supported for analysis. "
                    "Please use a dataset that inherits from BaseMapDataset."
                )
        else:
            # TODO: Implement HuggingFace Hub loading
            raise NotImplementedError(
                f"Dataset '{dataset_name}' is not registered in the REGISTRY. "
                "Loading from HuggingFace Hub is not yet implemented."
            )

    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_name}: {e}")
        raise


def _load_custom_dataset_from_path(
    dataset_path: str,
    dataset_format: Optional[str],
    tokenizer: Optional[Any],
    config: AnalyzeConfig,
) -> BaseMapDataset:
    """Load a custom dataset from a local file path.

    Args:
        dataset_path: Path to the dataset file
        dataset_format: Format of the dataset ('oumi' or 'alpaca') - required for
            custom datasets
        tokenizer: Optional tokenizer to use with the dataset
        config: Configuration object containing additional parameters

    Returns:
        Loaded dataset (TextSftJsonLinesDataset or VLJsonlinesDataset)
    """
    # Import here to avoid circular imports
    from oumi.datasets.sft.sft_jsonlines import TextSftJsonLinesDataset
    from oumi.datasets.vision_language.vision_jsonlines import VLJsonlinesDataset

    path = Path(dataset_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    if not path.is_file():
        raise ValueError(
            f"Dataset path must be a file, not a directory: {dataset_path}"
        )

    # Multimodal handling is explicit via config.is_multimodal
    if config.is_multimodal is True:
        # Note: processor_name requirement is already validated in AnalyzeConfig
        dataset_kwargs = {
            "dataset_path": str(path),
            "tokenizer": tokenizer,
            "processor_name": config.processor_name,
            "processor_kwargs": config.processor_kwargs,
            "trust_remote_code": config.trust_remote_code,
        }
        dataset_kwargs = {k: v for k, v in dataset_kwargs.items() if v is not None}
        dataset = VLJsonlinesDataset(**dataset_kwargs)
        logger.info(f"Loaded vision-language dataset from: {dataset_path}")
        return dataset
    elif config.is_multimodal is False:
        # If explicitly forced to text, load as text-only
        dataset_kwargs = {
            "dataset_path": str(path),
            "format": dataset_format,
        }
        if tokenizer is not None:
            dataset_kwargs["tokenizer"] = tokenizer
        dataset_kwargs = {k: v for k, v in dataset_kwargs.items() if v is not None}
        dataset = TextSftJsonLinesDataset(**dataset_kwargs)
        logger.info(f"Loaded text dataset from: {dataset_path}")
        return dataset
    else:
        # This should never happen due to config validation
        # is_multimodal=None case is already caught by AnalyzeConfig.__post_init__
        raise ValueError("Invalid vision-language configuration")


def compute_statistics(series: pd.Series, decimal_precision: int = 2) -> dict[str, Any]:
    """Compute statistics for a pandas Series.

    This utility function handles edge cases like empty series or single-element
    series, ensuring that standard deviation is 0.0 for single values instead
    of NaN.

    Args:
        series: Pandas Series containing numeric values
        decimal_precision: Number of decimal places for rounding

    Returns:
        Dictionary with computed statistics (count, mean, std, min, max, median)
    """
    if series.empty:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0,
            "max": 0,
            "median": 0.0,
        }

    if len(series) == 1:
        single_value = round(float(series.iloc[0]), decimal_precision)
        return {
            "count": 1,
            "mean": single_value,
            "std": 0.0,  # Standard deviation is 0 for single value
            "min": single_value,
            "max": single_value,
            "median": single_value,
        }

    return {
        "count": len(series),
        "mean": round(series.mean(), decimal_precision),
        "std": round(series.std(), decimal_precision),
        "min": round(series.min(), decimal_precision),
        "max": round(series.max(), decimal_precision),
        "median": round(series.median(), decimal_precision),
    }
