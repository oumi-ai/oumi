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

from typing import Optional, Union

import datasets

from oumi.core.configs import AnalyzeConfig
from oumi.core.datasets import BaseMapDataset
from oumi.core.registry import REGISTRY
from oumi.utils.logging import logger


def load_dataset_from_config(config: AnalyzeConfig) -> BaseMapDataset:
    """Load dataset based on configuration.

    This function loads datasets directly from the registry for analysis purposes.
    If a tokenizer is provided in the config, it will be passed to the dataset
    constructor.
    """
    dataset_name = config.dataset_name
    split = config.split
    subset = config.subset
    tokenizer = config.tokenizer

    if not dataset_name:
        raise ValueError("Dataset name is required")

    try:
        # Load dataset from the REGISTRY
        dataset_class = REGISTRY.get_dataset(dataset_name, subset=subset)

        if dataset_class is not None:
            # Prepare dataset constructor arguments
            dataset_kwargs = {
                "dataset_name": dataset_name,
                "dataset_path": None,
                "split": split,
                "subset": subset,
                "trust_remote_code": getattr(config, "trust_remote_code", False),
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
            # Load from HuggingFace Hub
            return _load_dataset_from_huggingface_hub(config)

    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_name}: {e}")
        raise


def _detect_vision_language_fields(sample: dict) -> dict[str, Union[str, None]]:
    """Detect appropriate field names for vision-language datasets.

    Args:
        sample: A sample from the dataset to analyze.

    Returns:
        Dictionary mapping field types to detected column names.
    """
    available_fields = list(sample.keys())

    # Common field name patterns for different components
    # For image fields, look for exact matches first, then patterns
    image_fields = []
    for f in available_fields:
        if f == "image":  # Exact match
            image_fields.insert(0, f)  # Prioritize exact match
        elif (
            "image" in f.lower()
            and "code" not in f.lower()
            and "synthesis" not in f.lower()
        ):
            # Image field but not code-related
            image_fields.append(f)

    text_fields = [
        f
        for f in available_fields
        if any(
            keyword in f.lower()
            for keyword in [
                "question",
                "prompt",
                "instruction",
                "text",
                "input",
                "caption",
            ]
        )
    ]
    answer_fields = [
        f
        for f in available_fields
        if any(
            keyword in f.lower()
            for keyword in ["answer", "output", "response", "solution", "label"]
        )
    ]

    # Select the most appropriate fields
    image_column = image_fields[0] if image_fields else None
    question_column = text_fields[0] if text_fields else None
    answer_column = answer_fields[0] if answer_fields else None

    if not image_column:
        raise ValueError(
            f"No image field found in dataset. Available fields: {available_fields}"
        )
    if not question_column:
        raise ValueError(
            f"No text/question field found in dataset. "
            f"Available fields: {available_fields}"
        )

    return {
        "image_column": image_column,
        "question_column": question_column,
        "answer_column": answer_column,
    }


def _load_dataset_from_huggingface_hub(config: AnalyzeConfig) -> BaseMapDataset:
    """Load dataset from HuggingFace Hub using existing infrastructure.

    This function provides robust loading with:
    - Automatic subset discovery and handling
    - Graceful fallback when subset loading fails
    - Comprehensive error handling and logging
    - Support for multiple dataset types (vision, text)
    """
    try:
        dataset_name = config.dataset_name
        split = config.split
        subset = config.subset

        # Ensure dataset_name is a string
        if not dataset_name:
            raise ValueError("Dataset name is required")

        logger.info(f"Loading dataset '{dataset_name}' from HuggingFace Hub")
        if subset:
            logger.info(f"Using subset: {subset}")

        # Handle subset loading with fallback
        temp_dataset = _load_dataset_with_subset_fallback(
            dataset_name, subset, split, config.trust_remote_code
        )

        # Get the first sample to analyze the structure
        # Handle both Dataset and IterableDataset
        if isinstance(temp_dataset, datasets.Dataset):
            sample = temp_dataset[0]
        else:
            # For IterableDataset, get the first item
            sample = next(iter(temp_dataset))

        logger.info(f"Dataset structure analysis: {list(sample.keys())}")

        has_image = "image" in sample
        has_messages = "messages" in sample

        # Determine which existing loader to use based on dataset structure
        if has_image:
            logger.info("Detected vision-language dataset")
            # Vision-language dataset - use HuggingFaceVisionDataset
            dataset_class = REGISTRY.get_dataset("hf_vision")
            if dataset_class is None:
                raise RuntimeError("HuggingFaceVisionDataset not found in registry")

            # Detect appropriate field names dynamically
            field_mapping = _detect_vision_language_fields(sample)
            logger.info(f"Detected field mapping: {field_mapping}")

            # Create dataset with detected vision-language parameters
            dataset = dataset_class(
                hf_dataset_path=dataset_name,
                image_column=field_mapping["image_column"],
                question_column=field_mapping["question_column"],
                answer_column=field_mapping["answer_column"],
                dataset_name=dataset_name,
                dataset_path=None,
                split=split,
                subset=subset,
                tokenizer=config.tokenizer,
                processor_name=config.processor_name,
                processor_kwargs=config.processor_kwargs,
                trust_remote_code=config.trust_remote_code,
            )

        elif has_messages:
            logger.info("Detected text dataset with messages")
            # Text dataset with messages - use HuggingFaceDataset
            dataset_class = REGISTRY.get_dataset("HuggingFaceDataset")
            if dataset_class is None:
                raise RuntimeError("HuggingFaceDataset not found in registry")

            # Create dataset with text parameters
            dataset = dataset_class(
                hf_dataset_path=dataset_name,
                messages_column="messages",
                dataset_name=dataset_name,
                dataset_path=None,
                split=split,
                subset=subset,
                tokenizer=config.tokenizer,
                trust_remote_code=config.trust_remote_code,
            )

        else:
            raise ValueError(
                f"Dataset '{dataset_name}' has unsupported structure. "
                f"Expected 'image' or 'messages' field, but found: "
                f"{list(sample.keys())}"
            )

        logger.info(f"Successfully loaded dataset '{dataset_name}'")
        return dataset

    except ImportError:
        raise ImportError(
            "The 'datasets' library is required to load datasets from HuggingFace Hub. "
            "Please install it with: pip install datasets"
        )
    except Exception as e:
        logger.error(
            f"Failed to load dataset '{dataset_name}' from HuggingFace Hub: {e}"
        )
        raise RuntimeError(
            f"Failed to load dataset '{dataset_name}' from HuggingFace Hub: {e}"
        )


def _load_dataset_with_subset_fallback(
    dataset_name: str,
    subset: Optional[str],
    split: str,
    trust_remote_code: bool,
) -> Union[
    datasets.Dataset,
    datasets.IterableDataset,
    datasets.DatasetDict,
    datasets.IterableDatasetDict,
]:
    """Load dataset with robust subset handling and fallback.

    Args:
        dataset_name: Name of the dataset
        subset: Subset name (can be None)
        split: Dataset split to load
        trust_remote_code: Whether to trust remote code

    Returns:
        Loaded dataset
    """
    try:
        # Try loading with subset first
        if subset:
            logger.info(f"Attempting to load subset '{subset}' from '{dataset_name}'")
            return datasets.load_dataset(
                path=dataset_name,
                name=subset,
                split=split,
                trust_remote_code=trust_remote_code,
            )
        else:
            logger.info(f"Loading dataset '{dataset_name}' without subset")
            return datasets.load_dataset(
                path=dataset_name,
                split=split,
                trust_remote_code=trust_remote_code,
            )
    except Exception as e:
        if subset:
            logger.warning(
                f"Failed to load subset '{subset}' from '{dataset_name}': {e}. "
                "Attempting to load without subset."
            )
            try:
                return datasets.load_dataset(
                    path=dataset_name,
                    split=split,
                    trust_remote_code=trust_remote_code,
                )
            except Exception as fallback_error:
                logger.error(
                    f"Failed to load dataset '{dataset_name}' even without subset: "
                    f"{fallback_error}"
                )
                raise fallback_error
        else:
            # Re-raise the original error if no subset was specified
            raise e
