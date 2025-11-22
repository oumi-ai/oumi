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

"""Validation utilities for prompt optimization datasets and configurations."""

from pathlib import Path
from typing import Any

from oumi.utils.logging import logger


class DatasetValidationError(ValueError):
    """Exception raised for invalid datasets."""

    pass


class ConfigValidationError(ValueError):
    """Exception raised for invalid configurations."""

    pass


def validate_dataset_file(
    dataset_path: str, name: str = "dataset", allow_missing: bool = False
) -> Path:
    """Validate that a dataset file exists and is readable.

    Args:
        dataset_path: Path to the dataset file.
        name: Name of the dataset (for error messages).
        allow_missing: If True, None is returned for missing files instead of raising.

    Returns:
        Path object for the dataset file.

    Raises:
        DatasetValidationError: If file doesn't exist or isn't readable.
    """
    if dataset_path is None:
        if allow_missing:
            return None  # type: ignore[return-value]
        raise DatasetValidationError(f"{name} path is required but was not provided")

    path = Path(dataset_path)

    if not path.exists():
        raise DatasetValidationError(
            f"{name} file not found: {dataset_path}\n"
            f"Please ensure the file exists and the path is correct."
        )

    if not path.is_file():
        raise DatasetValidationError(
            f"{name} path is not a file: {dataset_path}\n"
            f"Expected a JSONL file with one example per line."
        )

    if path.suffix.lower() not in [".jsonl", ".json"]:
        logger.warning(
            f"{name} file does not have .jsonl or .json extension: {dataset_path}. "
            f"Proceeding anyway, but ensure it's in JSONL format."
        )

    # Check if file is readable
    try:
        with open(path) as f:
            f.read(1)
    except OSError as e:
        raise DatasetValidationError(
            f"Cannot read {name} file: {dataset_path}\nError: {e}"
        )

    return path


def validate_dataset_example(
    example: dict[str, Any], line_num: int, dataset_name: str = "dataset"
) -> tuple[str, str]:
    """Validate a single dataset example and extract input/output.

    Args:
        example: Dictionary containing the example.
        line_num: Line number in the file (for error messages).
        dataset_name: Name of the dataset (for error messages).

    Returns:
        Tuple of (input_text, output_text).

    Raises:
        DatasetValidationError: If example is invalid.
    """
    if not isinstance(example, dict):
        raise DatasetValidationError(
            f"{dataset_name} line {line_num}: Expected a JSON object, got "
            f"{type(example).__name__}"
        )

    # Check for simple format: {"input": "...", "output": "..."}
    if "input" in example and "output" in example:
        input_text = example["input"]
        output_text = example["output"]

        if not isinstance(input_text, str):
            raise DatasetValidationError(
                f"{dataset_name} line {line_num}: 'input' must be a string, "
                f"got {type(input_text).__name__}"
            )

        if not isinstance(output_text, str):
            raise DatasetValidationError(
                f"{dataset_name} line {line_num}: 'output' must be a string, "
                f"got {type(output_text).__name__}"
            )

        if not input_text.strip():
            raise DatasetValidationError(
                f"{dataset_name} line {line_num}: 'input' cannot be empty"
            )

        if not output_text.strip():
            logger.warning(
                f"{dataset_name} line {line_num}: 'output' is empty. "
                f"This may cause issues during optimization."
            )

        return input_text, output_text

    # Check for Conversation format: {"messages": [...]}
    elif "messages" in example:
        from oumi.core.types.conversation import Conversation, Role

        try:
            conv = Conversation(**example)
        except Exception as e:
            raise DatasetValidationError(
                f"{dataset_name} line {line_num}: Invalid Conversation format: {e}"
            )

        # Extract last user message as input and last assistant message as output
        input_text = None
        output_text = None

        for msg in conv.messages:
            if msg.role == Role.USER:
                input_text = (
                    msg.content if isinstance(msg.content, str) else str(msg.content)
                )  # type: ignore[assignment]
            elif msg.role == Role.ASSISTANT:
                output_text = (
                    msg.content if isinstance(msg.content, str) else str(msg.content)
                )  # type: ignore[assignment]

        if input_text is None:
            raise DatasetValidationError(
                f"{dataset_name} line {line_num}: No user message found in conversation"
            )

        if output_text is None:
            raise DatasetValidationError(
                f"{dataset_name} line {line_num}: No assistant message found "
                f"in conversation"
            )

        return input_text, output_text  # type: ignore[return-value]

    else:
        raise DatasetValidationError(
            f"{dataset_name} line {line_num}: Example must have either "
            f"('input', 'output') fields or 'messages' field.\n"
            f"Found fields: {list(example.keys())}\n\n"
            f"Expected formats:\n"
            f'1. Simple: {{"input": "question here", "output": "answer here"}}\n'
            f'2. Conversation: {{"messages": [{{"role": "user", "content": '
            f'"..."}}, ...]}}'
        )


def validate_dataset_split_sizes(
    train_size: int, val_size: int, min_train: int = 10, min_val: int = 5
) -> None:
    """Validate that dataset splits have reasonable sizes.

    Args:
        train_size: Number of training examples.
        val_size: Number of validation examples.
        min_train: Minimum required training examples (default: 10).
        min_val: Minimum required validation examples (default: 5).

    Raises:
        DatasetValidationError: If split sizes are invalid.
    """
    # Stricter minimum requirements
    if train_size < min_train:
        raise DatasetValidationError(
            f"Training set has only {train_size} examples, but at least "
            f"{min_train} are required.\n"
            f"Prompt optimization needs sufficient examples to learn patterns.\n"
            f"Please provide more training data."
        )

    if val_size < min_val:
        raise DatasetValidationError(
            f"Validation set has only {val_size} examples, but at least "
            f"{min_val} are required.\n"
            f"Reliable evaluation requires sufficient validation examples.\n"
            f"Please provide more validation data or use a smaller "
            f"validation split."
        )

    # Error on extremely small datasets that will produce meaningless results
    if train_size < 20:
        raise DatasetValidationError(
            f"Training set has only {train_size} examples. "
            f"Prompt optimization requires at least 20 training examples for "
            f"meaningful results.\n"
            f"Most optimizers work best with 50+ examples.\n"
            f"Please provide more training data or consider using "
            f"pre-optimized prompts."
        )

    if val_size < 10:
        raise DatasetValidationError(
            f"Validation set has only {val_size} examples. "
            f"Evaluation requires at least 10 validation examples for "
            f"reliable scores.\n"
            f"Please provide more validation data or adjust your train/val "
            f"split ratio."
        )

    # Warn if sizes are small but acceptable
    if train_size < 50:
        logger.warning(
            f"Training set has only {train_size} examples. "
            f"Most optimizers work best with 50+ examples. "
            f"Consider providing more data for better optimization results."
        )

    if val_size < 20:
        logger.warning(
            f"Validation set has only {val_size} examples. "
            f"Evaluation scores may have higher variance with small validation sets. "
            f"Consider using 20+ validation examples for more reliable evaluation."
        )

    # Check ratio
    total = train_size + val_size
    val_ratio = val_size / total
    if val_ratio > 0.5:
        logger.warning(
            f"Validation set is {val_ratio:.1%} of total data. "
            f"Typically validation should be 10-20% of the data. "
            f"Consider using more data for training."
        )


def validate_optimizer_config(optimizer: str, num_trials: int) -> None:
    """Validate optimizer-specific configuration.

    Args:
        optimizer: Name of the optimizer.
        num_trials: Number of optimization trials.

    Raises:
        ConfigValidationError: If configuration is invalid for the optimizer.
    """
    optimizer = optimizer.lower()

    # Check for deprecated optimizers
    if optimizer == "evolutionary":
        raise ConfigValidationError(
            "The 'evolutionary' optimizer is deprecated and not functional.\n\n"
            "Please use one of these optimizers instead:\n"
            "  • 'mipro': Best for large datasets (300+ examples)\n"
            "  • 'gepa': Best for complex tasks with reflective optimization\n"
            "  • 'bootstrap': Best for small datasets (10-50 examples)\n\n"
            "See documentation for more details on each optimizer."
        )

    # Optimizer-specific validation
    if optimizer == "mipro":
        if num_trials < 10:
            logger.warning(
                f"MIPRO optimizer configured with only {num_trials} trials. "
                f"For best results, use at least 30 trials. "
                f"With few trials, consider using 'bootstrap' optimizer instead."
            )
        elif num_trials > 200:
            logger.warning(
                f"MIPRO optimizer configured with {num_trials} trials. "
                f"This may take a very long time and use many tokens. "
                f"Consider starting with 50-100 trials."
            )

    elif optimizer == "gepa":
        if num_trials < 5:
            logger.warning(
                f"GEPA optimizer configured with only {num_trials} trials. "
                f"GEPA works best with at least 10 trials."
            )

    elif optimizer == "bootstrap":
        if num_trials > 50:
            logger.warning(
                f"BootstrapFewShot optimizer configured with {num_trials} trials. "
                f"This optimizer is designed for quick few-shot selection "
                f"and typically doesn't need more than 20-30 trials."
            )
