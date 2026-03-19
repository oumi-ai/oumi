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

"""Unified training interface with simplified API."""

from __future__ import annotations

from dataclasses import fields
from typing import Any, TypeVar

from oumi.core.configs import (
    DataParams,
    DatasetParams,
    DatasetSplitParams,
    ModelParams,
    TrainerType,
    TrainingConfig,
    TrainingParams,
)
from oumi.core.configs.params.base_params import BaseParams

# =============================================================================
# Helper functions
# =============================================================================

T = TypeVar("T", bound=BaseParams)


def _merge_params(
    config_class: type[T],
    config_obj: T | None,
    flat_overrides: dict[str, Any],
) -> T:
    """Merge flat params over config object, using dataclass defaults for missing.

    Priority order: flat_overrides > config_obj > dataclass defaults

    Args:
        config_class: The dataclass type to create (e.g., TrainingParams)
        config_obj: Optional existing config object to use as base
        flat_overrides: Dict of flat param names to values (None values ignored)

    Returns:
        New instance of config_class with merged values
    """
    kwargs: dict[str, Any] = {}

    for field in fields(config_class):
        field_name = field.name
        flat_value = flat_overrides.get(field_name)

        if flat_value is not None:
            # Flat param provided and not None - use it
            kwargs[field_name] = flat_value
        elif config_obj is not None:
            # Use value from config object
            kwargs[field_name] = getattr(config_obj, field_name)
        # else: let dataclass use its default

    return config_class(**kwargs)


def _resolve_trainer_type(trainer_type: str) -> TrainerType:
    """Resolve trainer type string to TrainerType enum.

    Supports:
    - Exact enum value match (case-insensitive): "trl_sft" -> TRL_SFT
    - Short form via suffix matching: "sft" -> TRL_SFT (if unambiguous)

    Args:
        trainer_type: User-provided trainer type string

    Returns:
        TrainerType enum value

    Raises:
        ValueError: If trainer_type is not recognized or is ambiguous
    """
    trainer_type_lower = trainer_type.lower()

    # First, try exact match by enum value
    for tt in TrainerType:
        if tt.value == trainer_type_lower:
            return tt

    # Second, try short form matching (e.g., "sft" -> "trl_sft")
    matches = []
    for tt in TrainerType:
        # Check if the enum value ends with the provided string
        if tt.value.endswith(trainer_type_lower):
            matches.append(tt)
        # Also check if the enum name ends with it (case-insensitive)
        elif tt.name.lower().endswith(trainer_type_lower):
            matches.append(tt)

    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        match_names = [m.value for m in matches]
        raise ValueError(
            f"Ambiguous trainer_type: {trainer_type!r}. "
            f"Matches multiple types: {match_names}. "
            "Please use the full trainer type name."
        )

    # No match found
    supported = [tt.value for tt in TrainerType]
    raise ValueError(
        f"Unknown trainer_type: {trainer_type!r}. Supported types: {supported}"
    )


# =============================================================================
# Main class
# =============================================================================


class Trainer:
    """Unified training interface with simplified API.

    This class provides a simplified interface for training models by specifying
    a trainer type, model, and dataset, with optional configuration through flat
    parameters or full config objects.

    The precedence for parameters is: flat params > config objects > defaults.

    Examples:
        Simple usage with flat parameters:

        >>> from oumi.training import Trainer
        >>> trainer = Trainer(
        ...     trainer_type="sft",
        ...     model="meta-llama/Llama-2-7b",
        ...     dataset="yahma/alpaca-cleaned",
        ...     learning_rate=2e-4,
        ...     max_steps=100,
        ...     output_dir="/tmp/training_output",
        ... )
        >>> trainer.train()

        Using multiple datasets:

        >>> trainer = Trainer(
        ...     trainer_type="sft",
        ...     model="gpt2",
        ...     datasets=["dataset1", "dataset2"],
        ...     learning_rate=2e-4,
        ... )

        Advanced usage with config objects:

        >>> from oumi.core.configs import ModelParams, TrainingParams
        >>> trainer = Trainer(
        ...     trainer_type="dpo",
        ...     model_params=ModelParams(model_name="gpt2"),
        ...     training_params=TrainingParams(learning_rate=1e-4),
        ... )

        Mixing flat params and config objects (flat params override):

        >>> trainer = Trainer(
        ...     trainer_type="sft",
        ...     model_params=ModelParams(model_name="gpt2"),
        ...     learning_rate=5e-5,  # Overrides any learning_rate in training_params
        ... )

    Attributes:
        config: The built TrainingConfig object.
    """

    @classmethod
    def supported_trainer_types(cls) -> list[str]:
        """Return list of supported trainer type names."""
        return [tt.value for tt in TrainerType]

    def __init__(
        self,
        trainer_type: str,
        model: str | None = None,
        *,
        # --- Dataset params ---
        dataset: str | None = None,
        datasets: list[str] | None = None,
        eval_dataset: str | None = None,
        # --- Training core params ---
        learning_rate: float | None = None,
        num_train_epochs: int | None = None,
        max_steps: int | None = None,
        per_device_train_batch_size: int | None = None,
        output_dir: str | None = None,
        use_peft: bool | None = None,
        # --- Training extended params ---
        gradient_accumulation_steps: int | None = None,
        warmup_ratio: float | None = None,
        save_steps: int | None = None,
        logging_steps: int | None = None,
        eval_strategy: str | None = None,
        # --- Full config objects (flat params override these) ---
        training_config: TrainingConfig | None = None,
        model_params: ModelParams | None = None,
        training_params: TrainingParams | None = None,
        data_params: DataParams | None = None,
    ):
        """Initialize the unified trainer.

        Args:
            trainer_type: The trainer type to use. Call `supported_trainer_types()`
                for available options. Supports short forms like "sft" for "trl_sft".
            model: The model name/identifier. Required if model_params is not provided.

            dataset: Single dataset name/path for training.
            datasets: List of dataset names/paths for training mixtures.
            eval_dataset: Dataset name/path for evaluation.

            learning_rate: The initial learning rate for the optimizer.
            num_train_epochs: Total number of training epochs.
            max_steps: Maximum number of training steps (overrides num_train_epochs).
            per_device_train_batch_size: Batch size per device during training.
            output_dir: Directory where output files will be saved.
            use_peft: Whether to use Parameter-Efficient Fine-Tuning (PEFT).

            gradient_accumulation_steps: Number of steps to accumulate gradients.
            warmup_ratio: Ratio of total training steps for learning rate warmup.
            save_steps: Save a checkpoint every N training steps.
            logging_steps: Log every N training steps.
            eval_strategy: Evaluation strategy ("no", "steps", "epoch").

            training_config: Full TrainingConfig object. Flat params override.
            model_params: Full ModelParams config object. Flat params override.
            training_params: Full TrainingParams config object. Flat params override.
            data_params: Full DataParams config object. Flat params override.

        Raises:
            ValueError: If trainer_type is not supported, if model is not specified,
                or if both dataset and datasets are provided.
        """
        # Resolve trainer type
        self._trainer_type = _resolve_trainer_type(trainer_type)

        # Validate dataset arguments
        if dataset is not None and datasets is not None:
            raise ValueError(
                "Cannot specify both 'dataset' and 'datasets'. Use one or the other."
            )

        # Build the config
        self._config = self._build_config(
            model=model,
            dataset=dataset,
            datasets=datasets,
            eval_dataset=eval_dataset,
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            max_steps=max_steps,
            per_device_train_batch_size=per_device_train_batch_size,
            output_dir=output_dir,
            use_peft=use_peft,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_ratio=warmup_ratio,
            save_steps=save_steps,
            logging_steps=logging_steps,
            eval_strategy=eval_strategy,
            training_config=training_config,
            model_params=model_params,
            training_params=training_params,
            data_params=data_params,
        )

    def _build_config(
        self,
        model: str | None,
        dataset: str | None,
        datasets: list[str] | None,
        eval_dataset: str | None,
        learning_rate: float | None,
        num_train_epochs: int | None,
        max_steps: int | None,
        per_device_train_batch_size: int | None,
        output_dir: str | None,
        use_peft: bool | None,
        gradient_accumulation_steps: int | None,
        warmup_ratio: float | None,
        save_steps: int | None,
        logging_steps: int | None,
        eval_strategy: str | None,
        training_config: TrainingConfig | None,
        model_params: ModelParams | None,
        training_params: TrainingParams | None,
        data_params: DataParams | None,
    ) -> TrainingConfig:
        """Build the TrainingConfig from provided parameters."""
        # Extract base configs from training_config if provided
        base_model_params = training_config.model if training_config else model_params
        base_training_params = (
            training_config.training if training_config else training_params
        )
        base_data_params = training_config.data if training_config else data_params

        # Build ModelParams
        model_overrides: dict[str, Any] = {}
        if model is not None:
            model_overrides["model_name"] = model
        elif base_model_params is None or base_model_params.model_name is None:
            raise ValueError("Either 'model' or 'model_params' must be provided.")

        final_model_params = _merge_params(
            ModelParams, base_model_params, model_overrides
        )

        # Build TrainingParams
        training_overrides: dict[str, Any] = {
            "trainer_type": self._trainer_type,
            "learning_rate": learning_rate,
            "num_train_epochs": num_train_epochs,
            "max_steps": max_steps,
            "per_device_train_batch_size": per_device_train_batch_size,
            "output_dir": output_dir,
            "use_peft": use_peft,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "warmup_ratio": warmup_ratio,
            "save_steps": save_steps,
            "logging_steps": logging_steps,
            "eval_strategy": eval_strategy,
        }

        final_training_params = _merge_params(
            TrainingParams, base_training_params, training_overrides
        )

        # Build DataParams
        final_data_params = self._build_data_params(
            dataset=dataset,
            datasets=datasets,
            eval_dataset=eval_dataset,
            base_data_params=base_data_params,
        )

        # Build the final TrainingConfig
        if training_config is not None:
            # Start from the provided config and override
            config = TrainingConfig(
                model=final_model_params,
                training=final_training_params,
                data=final_data_params,
                peft=training_config.peft,
                fsdp=training_config.fsdp,
                deepspeed=training_config.deepspeed,
            )
        else:
            config = TrainingConfig(
                model=final_model_params,
                training=final_training_params,
                data=final_data_params,
            )

        return config

    def _build_data_params(
        self,
        dataset: str | None,
        datasets: list[str] | None,
        eval_dataset: str | None,
        base_data_params: DataParams | None,
    ) -> DataParams:
        """Build DataParams from dataset specifications."""
        # If no dataset args provided, use base_data_params or raise error
        if dataset is None and datasets is None:
            if base_data_params is not None:
                # Build eval if provided
                if eval_dataset is not None:
                    return DataParams(
                        train=base_data_params.train,
                        validation=DatasetSplitParams(
                            datasets=[DatasetParams(dataset_name=eval_dataset)]
                        ),
                        test=base_data_params.test,
                    )
                return base_data_params
            raise ValueError(
                "Either 'dataset', 'datasets', or 'data_params' must be provided."
            )

        # Build train datasets
        if dataset is not None:
            train_datasets = [DatasetParams(dataset_name=dataset)]
        else:
            assert datasets is not None
            train_datasets = [DatasetParams(dataset_name=d) for d in datasets]

        train_split = DatasetSplitParams(datasets=train_datasets)

        # Build validation datasets if provided
        validation_split = DatasetSplitParams()
        if eval_dataset is not None:
            validation_split = DatasetSplitParams(
                datasets=[DatasetParams(dataset_name=eval_dataset)]
            )

        return DataParams(
            train=train_split,
            validation=validation_split,
        )

    @property
    def config(self) -> TrainingConfig:
        """Access the built TrainingConfig."""
        return self._config

    @property
    def trainer_type(self) -> TrainerType:
        """The resolved trainer type."""
        return self._trainer_type

    def train(self, resume_from_checkpoint: str | None = None) -> dict | None:
        """Run training.

        Args:
            resume_from_checkpoint: Path to a checkpoint folder to resume from.
                If provided, training will resume from the specified checkpoint.

        Returns:
            Optional dictionary with training metrics if available.
        """
        from oumi.train import train

        # Update config if resume_from_checkpoint is provided
        if resume_from_checkpoint is not None:
            self._config.training.resume_from_checkpoint = resume_from_checkpoint

        return train(self._config)

    def __repr__(self) -> str:
        """Return a string representation of the trainer."""
        return (
            f"Trainer(trainer_type={self._trainer_type.value!r}, "
            f"model={self._config.model.model_name!r})"
        )
