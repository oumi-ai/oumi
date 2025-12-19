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

"""Generic SFT dataset using format converters.

This module provides GenericSftDataset, a flexible dataset class that uses
registered format converters to convert various data formats into Conversations.

Example usage:
    # With explicit converter
    dataset = GenericSftDataset(
        dataset_name="yahma/alpaca-cleaned",
        converter="alpaca",
    )

    # With auto-detection
    dataset = GenericSftDataset(
        dataset_path="/path/to/data.jsonl",
        converter="auto",  # or omit for auto-detection
    )

    # With converter factory kwargs
    dataset = GenericSftDataset(
        dataset_name="yahma/alpaca-cleaned",
        converter="alpaca",
        converter_kwargs={"include_system_prompt": True},
    )
"""

from typing import Any, Optional, Union

import pandas as pd
from typing_extensions import override

from oumi.core.converters.format_converters import (
    ConverterFn,
    auto_detect_converter,
    create_alpaca_converter,
)
from oumi.core.datasets import BaseSftDataset
from oumi.core.registry import REGISTRY, RegistryType, register_dataset
from oumi.core.types.conversation import Conversation
from oumi.utils.logging import logger


@register_dataset("generic_sft")
class GenericSftDataset(BaseSftDataset):
    """Generic SFT dataset that uses format converters for data conversion.

    This dataset class provides a flexible way to load data from various sources
    (HuggingFace Hub, local files) and convert them to Conversations using
    registered format converters.

    Unlike format-specific dataset classes (like AlpacaDataset), GenericSftDataset
    separates the data loading from format conversion, allowing users to specify
    the format converter explicitly or rely on auto-detection.

    Args:
        dataset_name: Name of the HuggingFace dataset or identifier.
        dataset_path: Path to local dataset file.
        converter: Name of the format converter to use. Options:
            - "oumi": For {"messages": [{"role": ..., "content": ...}]}
            - "alpaca": For {"instruction": ..., "input": ..., "output": ...}
            - "sharegpt": For {"conversations": [{"from": ..., "value": ...}]}
            - "langfuse": For Langfuse export format
            - "opentelemetry": For OpenTelemetry LLM spans
            - "langchain": For LangChain traces
            - "auto" or None: Auto-detect from data structure
        converter_kwargs: Arguments for converter factories (e.g., Alpaca's system prompt).
        **kwargs: Additional arguments passed to BaseSftDataset.

    Example:
        Loading from HuggingFace with explicit converter:
            >>> dataset = GenericSftDataset(
            ...     dataset_name="yahma/alpaca-cleaned",
            ...     converter="alpaca",
            ... )

        Loading from local file with auto-detection:
            >>> dataset = GenericSftDataset(
            ...     dataset_path="/path/to/data.jsonl",
            ... )

        Using converter with custom configuration:
            >>> dataset = GenericSftDataset(
            ...     dataset_name="yahma/alpaca-cleaned",
            ...     converter="alpaca",
            ...     converter_kwargs={"include_system_prompt": True},
            ... )
    """

    default_dataset = "generic_sft"

    def __init__(
        self,
        *,
        converter: Optional[str] = None,
        converter_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """Initialize GenericSftDataset.

        Args:
            converter: Name of format converter to use. If None or "auto",
                auto-detects from data structure.
            converter_kwargs: Arguments for converter factory functions.
            **kwargs: Additional arguments for BaseSftDataset.
        """
        self._converter_name: Optional[str] = converter
        self._converter_kwargs: dict[str, Any] = converter_kwargs or {}
        self._converter_fn: Optional[ConverterFn] = None

        # Call parent init which will call _load_data
        super().__init__(**kwargs)

        # Initialize converter after data is loaded (needed for auto-detection)
        self._init_converter()

    def _init_converter(self) -> None:
        """Initialize the converter function based on configuration or auto-detection."""
        converter_name = self._converter_name

        # Auto-detect if converter not specified or set to "auto"
        if converter_name is None or converter_name == "auto":
            if len(self._data) == 0:
                raise ValueError(
                    "Cannot auto-detect converter: dataset is empty. "
                    "Please specify a converter explicitly."
                )
            # Get first example for auto-detection
            first_row = self._data.iloc[0]
            example = (
                first_row.to_dict() if hasattr(first_row, "to_dict") else dict(first_row)
            )
            converter_name = auto_detect_converter(example)
            logger.info(f"Auto-detected format converter: '{converter_name}'")

        self._converter_name = converter_name

        # Handle special factory converters with kwargs
        if self._converter_kwargs:
            if converter_name == "alpaca":
                # Use the factory for alpaca with custom settings
                self._converter_fn = create_alpaca_converter(**self._converter_kwargs)
                logger.info(
                    f"Created alpaca converter with kwargs: {self._converter_kwargs}"
                )
                return
            else:
                logger.warning(
                    f"converter_kwargs provided but converter '{converter_name}' "
                    "may not support factory-style creation. Attempting to use kwargs."
                )

        # Get converter from registry
        converter = REGISTRY.get_converter(converter_name)
        if converter is None:
            available = list(REGISTRY.get_all(RegistryType.CONVERTER).keys())
            raise ValueError(
                f"Unknown converter: '{converter_name}'. "
                f"Available converters: {available}"
            )

        # If kwargs provided and converter might be a factory, try calling it
        if self._converter_kwargs:
            try:
                self._converter_fn = converter(**self._converter_kwargs)
                logger.info(
                    f"Created converter '{converter_name}' with kwargs: "
                    f"{self._converter_kwargs}"
                )
                return
            except TypeError:
                # Not a factory, use as-is
                logger.debug(
                    f"Converter '{converter_name}' is not a factory, "
                    "ignoring converter_kwargs"
                )

        self._converter_fn = converter

    @override
    def transform_conversation(self, example: Union[dict, pd.Series]) -> Conversation:
        """Transform a data example into a Conversation using the configured converter.

        Args:
            example: The raw data example (dict or pandas Series).

        Returns:
            Conversation object.

        Raises:
            RuntimeError: If converter is not initialized.
            ValueError: If the example cannot be converted.
        """
        if self._converter_fn is None:
            raise RuntimeError(
                "Converter not initialized. This is an internal error - "
                "please report it."
            )

        # Convert pandas Series to dict if needed
        if isinstance(example, pd.Series):
            example = example.to_dict()

        try:
            return self._converter_fn(example)
        except Exception as e:
            raise ValueError(
                f"Failed to convert example with '{self._converter_name}' converter: "
                f"{str(e)}"
            ) from e

    @property
    def converter_name(self) -> Optional[str]:
        """Get the name of the converter being used."""
        return self._converter_name
