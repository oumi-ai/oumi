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
from typing import Any, Optional, Union

import pandas as pd
from tqdm import tqdm

from oumi.builders.models import build_tokenizer
from oumi.core.configs.analyze_config import AnalyzeConfig
from oumi.core.configs.params.model_params import ModelParams
from oumi.core.datasets.base_iterable_dataset import BaseIterableDataset
from oumi.core.datasets.base_map_dataset import BaseMapDataset
from oumi.core.registry.registry import REGISTRY
from oumi.core.types.conversation import Conversation

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
) -> Union[BaseMapDataset, BaseIterableDataset]:
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
            # Check if this is an iterable dataset that supports streaming
            import inspect

            from oumi.core.datasets.base_iterable_dataset import BaseIterableDataset

            # Ensure dataset_class is actually a class before using issubclass
            is_iterable_dataset = inspect.isclass(dataset_class) and issubclass(
                dataset_class, BaseIterableDataset
            )

            # For iterable datasets, force streaming mode to avoid downloading all
            if is_iterable_dataset:
                logger.info(
                    f"Using streaming mode for iterable dataset: {dataset_name}"
                )
                # Don't modify split for iterable datasets - streaming handles limiting
            else:
                # For map datasets, modify split to include slicing if sample_count set
                if config.sample_count is not None and config.sample_count > 0:
                    # Use a larger slice (10x sample_count) to ensure enough data
                    # after any filtering that might happen in the dataset
                    slice_size = config.sample_count * 10
                    if "[" not in split:  # Only add slicing if not already present
                        split = f"{split}[:{slice_size}]"

            # Prepare dataset constructor arguments
            dataset_kwargs = {
                "dataset_name": dataset_name,
                "dataset_path": None,
                "split": split,
                "subset": subset,
                "trust_remote_code": config.trust_remote_code,
            }

            # Force streaming for iterable datasets
            if is_iterable_dataset:
                dataset_kwargs["stream"] = True

            # Add tokenizer if provided
            if tokenizer is not None:
                dataset_kwargs["tokenizer"] = tokenizer

            # Add processor parameters for vision-language datasets
            if config.processor_name:
                dataset_kwargs["processor_name"] = config.processor_name
                dataset_kwargs["processor_kwargs"] = config.processor_kwargs
                dataset_kwargs["trust_remote_code"] = config.trust_remote_code

            # Add required parameters for pretraining datasets
            if is_iterable_dataset:
                # Import here to avoid circular imports
                from oumi.core.datasets.base_pretraining_dataset import (
                    BasePretrainingDataset,
                )

                if inspect.isclass(dataset_class) and issubclass(
                    dataset_class, BasePretrainingDataset
                ):
                    # Pretraining datasets require seq_length and tokenizer
                    if "seq_length" not in dataset_kwargs:
                        dataset_kwargs["seq_length"] = 64  # Default sequence length
                    if tokenizer is None:
                        # Create a default tokenizer if none provided
                        from oumi.builders.models import build_tokenizer
                        from oumi.core.configs.params.model_params import ModelParams

                        model_params = ModelParams(
                            model_name="openai-community/gpt2",
                            tokenizer_kwargs={"pad_token": "<|endoftext|>"},
                        )
                        dataset_kwargs["tokenizer"] = build_tokenizer(model_params)

            # Load registered dataset with parameters
            # Check if dataset_class is callable (class or mock for testing)
            if not (inspect.isclass(dataset_class) or callable(dataset_class)):
                raise TypeError(
                    f"Expected class or callable, got {type(dataset_class)} for "
                    f"dataset {dataset_name}"
                )
            dataset = dataset_class(**dataset_kwargs)

            # Ensure we return a supported dataset type
            if isinstance(dataset, (BaseMapDataset, BaseIterableDataset)):
                return dataset
            else:
                raise NotImplementedError(
                    f"Dataset type {type(dataset)} is not supported for analysis. "
                    "Please use a dataset that inherits from BaseMapDataset or "
                    "BaseIterableDataset."
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


def conversation_to_dataframes(
    conversation: Conversation, conversation_id: str, conversation_idx: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert a single conversation to separate conversation and message DataFrames.

    This creates two DataFrames: one for conversation-level data and one for
    message-level data, suitable for comprehensive dataset analysis.

    Args:
        conversation: The conversation object to convert
        conversation_id: ID of the conversation
        conversation_idx: Index of the conversation

    Returns:
        Tuple of (conversation_df, message_df)
    """
    # Create conversation-level data
    conversation_data = {
        "conversation_index": conversation_idx,
        "conversation_id": conversation_id,
        "num_messages": len(conversation.messages),
    }
    conversation_df = pd.DataFrame([conversation_data])

    # Create message-level data
    messages_data = []
    for msg_idx, message in enumerate(conversation.messages):
        text_content = (
            message.content
            if isinstance(message.content, str)
            else message.compute_flattened_text_content()
        )
        messages_data.append(
            {
                "conversation_index": conversation_idx,
                "conversation_id": conversation_id,
                "message_index": msg_idx,
                "message_id": message.id or f"msg_{msg_idx}",
                "role": message.role.value,
                "text_content": text_content,
            }
        )

    message_df = pd.DataFrame(messages_data)
    return conversation_df, message_df


def convert_dataset_to_dataframes(
    dataset,  # Union[BaseMapDataset, BaseIterableDataset]
    items_to_analyze: int,
    dataset_name: str = "Dataset",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert a dataset to conversations and messages DataFrames.

    This method handles different dataset types:
    - SFT/Vision-SFT/GRPO datasets: Convert via conversation() method
    - DPO/KTO/Pretraining datasets: Convert raw data structure
    - Iterable datasets: Stream and limit to items_to_analyze

    Args:
        dataset: The dataset to process (BaseMapDataset or BaseIterableDataset)
        items_to_analyze: Number of items to analyze
        dataset_name: Name of the dataset for progress display

    Returns:
        Tuple of (conversations_df, messages_df) ready for analysis

    Raises:
        ValueError: If dataset is not provided
    """
    if dataset is None:
        raise ValueError("Dataset must be provided for conversation processing")

    # Handle iterable datasets (streaming datasets like C4)
    if isinstance(dataset, BaseIterableDataset):
        return _convert_iterable_dataset_to_dataframes(
            dataset, items_to_analyze, dataset_name
        )

    # Check if dataset has conversation() method (SFT/Vision-SFT/GRPO datasets)
    elif hasattr(dataset, "conversation") and callable(
        getattr(dataset, "conversation")
    ):
        return _convert_conversation_dataset_to_dataframes(
            dataset, items_to_analyze, dataset_name
        )
    else:
        # For non-conversation datasets (DPO, KTO, pretraining), convert raw data
        return _convert_raw_dataset_to_dataframes(
            dataset, items_to_analyze, dataset_name
        )


def _convert_iterable_dataset_to_dataframes(
    dataset: BaseIterableDataset, items_to_analyze: int, dataset_name: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert iterable datasets (streaming) to DataFrames.

    This handles datasets like C4, Pile, etc. that are streamed and don't support index.
    We iterate through the dataset and stop after items_to_analyze items.
    """
    from oumi.utils.logging import logger

    # For now, create empty conversations DataFrame since these datasets don't have
    # conversation structure
    conversations_df = pd.DataFrame()

    # Stream raw data and limit to items_to_analyze
    raw_data_list = []

    try:
        # Use tqdm to show progress while streaming
        dataset_iter = iter(dataset)
        for idx in tqdm(
            range(items_to_analyze),
            desc=f"Streaming {dataset_name} data",
            unit="item",
        ):
            try:
                # Get next item from the iterator
                raw_item = next(dataset_iter)

                # Convert to dict if it's not already
                if hasattr(raw_item, "to_dict") and callable(
                    getattr(raw_item, "to_dict")
                ):
                    raw_item = raw_item.to_dict()  # type: ignore
                elif not isinstance(raw_item, dict):
                    # For pretraining datasets, the item might be a tensor or list
                    # Convert to a simple dict structure
                    raw_item = {"input_ids": raw_item, "item_index": idx}

                # Add index information
                raw_item["item_index"] = idx
                raw_data_list.append(raw_item)

            except StopIteration:
                # Dataset ended before we reached items_to_analyze
                logger.info(
                    f"Dataset ended after {idx} items (requested {items_to_analyze})"
                )
                break
            except Exception as e:
                logger.warning(f"Failed to process item {idx} from {dataset_name}: {e}")
                continue

    except Exception as e:
        logger.error(f"Failed to iterate over dataset {dataset_name}: {e}")
        # Return empty DataFrames if we can't iterate
        return pd.DataFrame(), pd.DataFrame()

    # Create a DataFrame from raw data for analysis
    if raw_data_list:
        messages_df = pd.DataFrame(raw_data_list)
        # Add required columns for analysis compatibility
        messages_df["conversation_index"] = messages_df["item_index"]
        messages_df["message_index"] = (
            0  # Single message per item for non-conversation data
        )
        messages_df["message_id"] = messages_df["item_index"].apply(
            lambda x: f"msg_{x}"
        )
    else:
        messages_df = pd.DataFrame()

    return conversations_df, messages_df


def _convert_conversation_dataset_to_dataframes(
    dataset, items_to_analyze: int, dataset_name: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert datasets with conversation() method to DataFrames."""
    conversation_df_list = []
    message_df_list = []

    for conversation_idx in tqdm(
        range(items_to_analyze),
        desc=f"Converting {dataset_name} to DataFrames",
        unit="item",
    ):
        conversation = dataset.conversation(conversation_idx)
        conversation_id = conversation.conversation_id or str(conversation_idx)
        conversation_df, message_df = conversation_to_dataframes(
            conversation, conversation_id, conversation_idx
        )

        # Collect all DataFrames for concatenation
        if not conversation_df.empty:
            conversation_df_list.append(conversation_df)
        if not message_df.empty:
            message_df_list.append(message_df)

    # Create complete DataFrames by concatenating all individual DataFrames
    conversations_df = (
        pd.concat(conversation_df_list, ignore_index=True)
        if conversation_df_list
        else pd.DataFrame()
    )
    messages_df = (
        pd.concat(message_df_list, ignore_index=True)
        if message_df_list
        else pd.DataFrame()
    )

    return conversations_df, messages_df


def _convert_raw_dataset_to_dataframes(
    dataset, items_to_analyze: int, dataset_name: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert datasets without conversation() method to DataFrames.

    This handles DPO, KTO, and pretraining datasets that maintain their original.
    """
    from oumi.utils.logging import logger

    # For now, create empty conversations DataFrame since these datasets don't have
    # conversation structure
    conversations_df = pd.DataFrame()

    # Get raw data structure for analysis
    raw_data_list = []
    for idx in tqdm(
        range(items_to_analyze),
        desc=f"Converting {dataset_name} raw data",
        unit="item",
    ):
        try:
            # Get raw data from the dataset
            if hasattr(dataset, "raw"):
                raw_item = dataset.raw(idx)
            else:
                raw_item = dataset[idx]

            # Convert to dict if it's a pandas Series
            if hasattr(raw_item, "to_dict"):
                raw_item = raw_item.to_dict()

            # Add index information
            raw_item["item_index"] = idx
            raw_data_list.append(raw_item)

        except Exception as e:
            logger.warning(f"Failed to process item {idx} from {dataset_name}: {e}")
            continue

    # Create a DataFrame from raw data for analysis
    if raw_data_list:
        messages_df = pd.DataFrame(raw_data_list)
        # Add required columns for analysis compatibility
        messages_df["conversation_index"] = messages_df["item_index"]
        messages_df["message_index"] = (
            0  # Single message per item for non-conversation data
        )
        messages_df["message_id"] = messages_df["item_index"].apply(
            lambda x: f"msg_{x}"
        )
    else:
        messages_df = pd.DataFrame()

    return conversations_df, messages_df


def get_schema_for_format(dataset_format: str) -> dict:
    """Get column schema configuration based on dataset format.

    Args:
        dataset_format: Format of the dataset. Supported formats:
            - 'oumi': Conversation format (messages with roles)
            - 'alpaca': Instruction format (instruction/input/output)
            - 'prompt_response': Simple prompt/response pairs
            - 'dpo': Preference tuning format (prompt/chosen/rejected)
            - 'pretraining': Raw text for pretraining
            - 'kto': Binary feedback format (prompt/completion/label)

    Returns:
        Dictionary mapping column names to their configuration

    Raises:
        ValueError: If dataset_format is not supported
    """
    format_map = {
        "oumi": get_conversation_schema,
        "alpaca": get_alpaca_schema,
        "prompt_response": get_prompt_response_schema,
        "dpo": get_dpo_schema,
        "pretraining": get_pretraining_schema,
        "kto": get_kto_schema,
    }

    if dataset_format in format_map:
        return format_map[dataset_format]()
    else:
        raise ValueError(
            f"Unsupported dataset format: {dataset_format}. "
            f"Supported formats: {list(format_map.keys())}"
        )


def get_conversation_schema() -> dict:
    """Get column configuration for conversation format (oumi format).

    Returns:
        Dictionary mapping column names to their configuration.
    """
    from oumi.core.analyze.column_types import ColumnType, ContentType

    return {
        # Conversation DataFrame columns
        "conversation_index": {
            "type": ColumnType.INT,
            "content_type": ContentType.METADATA,
            "description": "Conversation index in dataset",
        },
        "conversation_id": {
            "type": ColumnType.STRING,
            "content_type": ContentType.METADATA,
            "description": "Conversation identifier",
        },
        "num_messages": {
            "type": ColumnType.INT,
            "content_type": ContentType.NUMERIC,
            "description": "Number of messages in conversation",
        },
        # Message DataFrame columns
        "message_index": {
            "type": ColumnType.INT,
            "content_type": ContentType.METADATA,
            "description": "Message index within conversation",
        },
        "message_id": {
            "type": ColumnType.STRING,
            "content_type": ContentType.METADATA,
            "description": "Message identifier",
        },
        "role": {
            "type": ColumnType.STRING,
            "content_type": ContentType.CATEGORICAL,
            "description": "Message role (user/assistant/system)",
        },
        "text_content": {
            "type": ColumnType.STRING,
            "content_type": ContentType.TEXT,
            "description": "Message text content",
        },
        # Additional fields that might be present
        "timestamp": {
            "type": ColumnType.TIMESTAMP,
            "content_type": ContentType.METADATA,
            "description": "Message timestamp",
        },
        "processing_time": {
            "type": ColumnType.FLOAT,
            "content_type": ContentType.NUMERIC,
            "description": "AI processing time in seconds",
        },
        "model": {
            "type": ColumnType.STRING,
            "content_type": ContentType.METADATA,
            "description": "Model used for generation",
        },
        "temperature": {
            "type": ColumnType.FLOAT,
            "content_type": ContentType.METADATA,
            "description": "Sampling temperature",
        },
        "max_tokens": {
            "type": ColumnType.INT,
            "content_type": ContentType.METADATA,
            "description": "Maximum tokens to generate",
        },
    }


def get_alpaca_schema() -> dict:
    """Get column configuration for alpaca format (instruction format).

    Returns:
        Dictionary mapping column names to their configuration.
    """
    from oumi.core.analyze.column_types import ColumnType, ContentType

    return {
        "instruction": {
            "type": ColumnType.STRING,
            "content_type": ContentType.TEXT,
            "description": "Instruction or prompt text",
        },
        "input": {
            "type": ColumnType.STRING,
            "content_type": ContentType.TEXT,
            "description": "Input context or additional information",
        },
        "output": {
            "type": ColumnType.STRING,
            "content_type": ContentType.TEXT,
            "description": "Expected output or response",
        },
        # Common additional fields
        "text": {
            "type": ColumnType.STRING,
            "content_type": ContentType.TEXT,
            "description": "Combined text content",
        },
    }


def get_prompt_response_schema() -> dict:
    """Get column configuration for prompt/response format.

    Returns:
        Dictionary mapping column names to their configuration.
    """
    from oumi.core.analyze.column_types import ColumnType, ContentType

    return {
        "prompt": {
            "type": ColumnType.STRING,
            "content_type": ContentType.TEXT,
            "description": "Input prompt or question",
        },
        "response": {
            "type": ColumnType.STRING,
            "content_type": ContentType.TEXT,
            "description": "Model response or answer",
        },
        # Common variations
        "instruction": {
            "type": ColumnType.STRING,
            "content_type": ContentType.TEXT,
            "description": "Instruction or prompt text",
        },
        "output": {
            "type": ColumnType.STRING,
            "content_type": ContentType.TEXT,
            "description": "Expected output or response",
        },
    }


def get_dpo_schema() -> dict:
    """Get column configuration for DPO (preference tuning) format.

    Returns:
        Dictionary mapping column names to their configuration.
    """
    from oumi.core.analyze.column_types import ColumnType, ContentType

    return {
        "prompt": {
            "type": ColumnType.STRING,
            "content_type": ContentType.TEXT,
            "description": "Input prompt for preference comparison",
        },
        "chosen": {
            "type": ColumnType.STRING,
            "content_type": ContentType.TEXT,
            "description": "Preferred response",
        },
        "rejected": {
            "type": ColumnType.STRING,
            "content_type": ContentType.TEXT,
            "description": "Rejected response",
        },
        # Metadata fields
        "score_chosen": {
            "type": ColumnType.FLOAT,
            "content_type": ContentType.NUMERIC,
            "description": "Score for chosen response",
        },
        "score_rejected": {
            "type": ColumnType.FLOAT,
            "content_type": ContentType.NUMERIC,
            "description": "Score for rejected response",
        },
    }


def get_pretraining_schema() -> dict:
    """Get column configuration for pretraining format.

    Returns:
        Dictionary mapping column names to their configuration.
    """
    from oumi.core.analyze.column_types import ColumnType, ContentType

    return {
        "text": {
            "type": ColumnType.STRING,
            "content_type": ContentType.TEXT,
            "description": "Raw text content for pretraining",
        },
        "content": {
            "type": ColumnType.STRING,
            "content_type": ContentType.TEXT,
            "description": "Text content (alternative field name)",
        },
        # Metadata fields common in pretraining datasets
        "url": {
            "type": ColumnType.STRING,
            "content_type": ContentType.METADATA,
            "description": "Source URL",
        },
        "timestamp": {
            "type": ColumnType.TIMESTAMP,
            "content_type": ContentType.METADATA,
            "description": "Content timestamp",
        },
        "source": {
            "type": ColumnType.STRING,
            "content_type": ContentType.CATEGORICAL,
            "description": "Data source identifier",
        },
        "language": {
            "type": ColumnType.STRING,
            "content_type": ContentType.CATEGORICAL,
            "description": "Content language",
        },
        "quality_score": {
            "type": ColumnType.FLOAT,
            "content_type": ContentType.NUMERIC,
            "description": "Content quality score",
        },
        "length": {
            "type": ColumnType.INT,
            "content_type": ContentType.NUMERIC,
            "description": "Content length in characters",
        },
    }


def get_kto_schema() -> dict:
    """Get column configuration for KTO (binary feedback) format.

    Returns:
        Dictionary mapping column names to their configuration.
    """
    from oumi.core.analyze.column_types import ColumnType, ContentType

    return {
        "prompt": {
            "type": ColumnType.STRING,
            "content_type": ContentType.TEXT,
            "description": "Input prompt",
        },
        "completion": {
            "type": ColumnType.STRING,
            "content_type": ContentType.TEXT,
            "description": "Model completion/response",
        },
        "label": {
            "type": ColumnType.BOOL,
            "content_type": ContentType.CATEGORICAL,
            "description": "Binary feedback label (True=desirable, False=undesirable)",
        },
        # Additional fields
        "score": {
            "type": ColumnType.FLOAT,
            "content_type": ContentType.NUMERIC,
            "description": "Numeric score for the completion",
        },
    }
