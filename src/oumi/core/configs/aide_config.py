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

from dataclasses import dataclass, field

from oumi.core.configs.base_config import BaseConfig
from oumi.core.configs.params.aide_params import AideParams
from oumi.core.configs.params.data_params import DataParams
from oumi.core.configs.params.model_params import ModelParams


@dataclass
class AideConfig(BaseConfig):
    """Top-level configuration for ``oumi aide``.

    Combines Oumi's standard model/data params with AIDE-specific settings
    to define a complete agentic optimization run. This parallels
    :class:`~oumi.core.configs.tuning_config.TuningConfig` for Optuna-based
    hyperparameter tuning.

    Example YAML::

        model:
          model_name: "HuggingFaceTB/SmolLM2-135M-Instruct"
        data:
          train:
            datasets:
              - dataset_name: "yahma/alpaca-cleaned"
        goal: "Minimize eval_loss by tuning hyperparameters"
        aide:
          steps: 10
          target_metric: "eval_loss"
          target_direction: "minimize"

    See Also:
        - :class:`oumi.core.configs.params.aide_params.AideParams`
        - :class:`oumi.core.configs.tuning_config.TuningConfig`
    """

    data: DataParams = field(default_factory=DataParams)
    """Parameters for the dataset.

    This field contains all the necessary settings for data processing and loading.
    It includes options for train and evaluation datasets and preprocessing steps.

    For more details, see the :class:`oumi.core.configs.params.data_params.DataParams`
    class.
    """

    model: ModelParams = field(default_factory=ModelParams)
    """Parameters for the model.

    This field defines the model architecture, size, and other model-specific settings.
    It includes options for model type, pretrained weights, and tokenizer configuration.

    For more details, see :class:`oumi.core.configs.params.model_params.ModelParams`
    class.
    """

    aide: AideParams = field(default_factory=AideParams)
    """Parameters for the AIDE agentic optimization process.

    This field includes settings for the search algorithm, LLM selection,
    optimization target, and execution sandbox.

    For more details, see :class:`oumi.core.configs.params.aide_params.AideParams`.
    """

    goal: str = ""
    """Natural language description of the optimization goal.

    This is included in the LLM prompt to guide code generation. Be specific
    about what you want to optimize and any constraints.

    Example: "Optimize training hyperparameters for SmolLM 135M on Alpaca
    to minimize eval_loss. Focus on learning rate and warmup schedule."
    """

    base_training_config: str | None = None
    """Path to a base Oumi training config YAML.

    For CONFIG_SEARCH surface: AIDE uses this as a starting point and
    generates modifications. If not provided, AIDE generates configs
    from scratch.
    """

    mutable_config_paths: list[str] = field(default_factory=list)
    """Config field paths that AIDE is allowed to modify.

    For CONFIG_SEARCH surface: constrains which training config fields
    the agent can change. Empty list means all fields are mutable.

    Example: ``["training.learning_rate", "training.optimizer", "peft.lora_r"]``
    """

    eval_task_names: list[str] = field(default_factory=list)
    """Names of custom evaluation functions to run after training.

    These should be registered via ``@register_evaluation_function``.
    If empty, the target metric is expected from the training eval results.
    """

    def finalize_and_validate(self) -> None:
        """Validates the AIDE config.

        Overrides base validation to make data params optional — AIDE
        generates its own training scripts so datasets are not always
        required at config time. Only validates data if datasets are
        actually specified.
        """
        # Validate aide params and their children
        self.aide.finalize_and_validate()
        self.model.finalize_and_validate()

        # Only validate data if datasets are explicitly provided
        has_train_data = (
            self.data.train
            and self.data.train.datasets
            and len(self.data.train.datasets) > 0
        )
        if has_train_data:
            self.data.finalize_and_validate()

        # Custom validation
        if not self.goal:
            raise ValueError(
                "A 'goal' description is required for AIDE optimization. "
                "Provide a natural language description of what to optimize."
            )
