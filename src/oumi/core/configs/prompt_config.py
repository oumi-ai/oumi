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
from oumi.core.configs.inference_engine_type import InferenceEngineType
from oumi.core.configs.params.generation_params import GenerationParams
from oumi.core.configs.params.model_params import ModelParams
from oumi.core.configs.params.prompt_optimization_params import (
    PromptOptimizationParams,
)
from oumi.core.configs.params.remote_params import RemoteParams


@dataclass
class PromptOptimizationConfig(BaseConfig):
    """Configuration for prompt optimization."""

    model: ModelParams = field(default_factory=ModelParams)
    """Parameters for the model used during prompt optimization."""

    generation: GenerationParams = field(default_factory=GenerationParams)
    """Parameters for text generation during optimization."""

    optimization: PromptOptimizationParams = field(
        default_factory=PromptOptimizationParams
    )
    """Parameters specific to prompt optimization."""

    train_dataset_path: str | None = None
    """Path to the training dataset file (JSONL format).

    Each line should be a JSON object with 'input' and 'output' fields,
    or an Oumi Conversation object.
    """

    val_dataset_path: str | None = None
    """Path to the validation dataset file (JSONL format).

    Used for evaluating optimized prompts. If not provided, a split from
    the training dataset will be used.
    """

    initial_prompt: str | None = None
    """Initial prompt/instruction to optimize.

    If not provided, the optimizer will generate an initial prompt.
    """

    output_dir: str = "./prompt_optimization_output"
    """Directory to save optimization results and artifacts."""

    metric: str = "accuracy"
    """Evaluation metric to optimize for.

    Supported metrics:
        - accuracy: Classification/exact match accuracy
        - f1: F1 score (token-level)
        - bleu: BLEU score for generation tasks
        - rouge: ROUGE score for summarization
        - embedding_similarity: Cosine similarity between embeddings (requires
          sentence-transformers)
        - bertscore: BERTScore for semantic similarity (requires bert-score)
        - llm_judge: Use an LLM to judge quality (requires openai, can be expensive)
        - custom: Use a custom metric function
    """

    custom_metric_path: str | None = None
    """Path to a Python file containing a custom metric function.

    The file should define a function named 'metric_fn' that takes
    predictions and references and returns a score between 0 and 1.
    """

    engine: InferenceEngineType | None = None
    """The inference engine to use for evaluation during optimization.

    This field is required.
    """

    remote_params: RemoteParams | None = None
    """Parameters for running inference against a remote API."""

    max_training_samples: int | None = None
    """Maximum number of training samples to use for optimization.

    If not set, all training samples will be used.
    """

    max_validation_samples: int | None = None
    """Maximum number of validation samples to use for evaluation.

    If not set, all validation samples will be used.
    """

    def __finalize_and_validate__(self) -> None:
        """Validates the prompt optimization configuration."""
        if self.train_dataset_path is None:
            raise ValueError("train_dataset_path must be specified")

        if self.engine is None:
            raise ValueError(
                "engine must be specified for prompt optimization inference"
            )

        valid_metrics = {
            "accuracy",
            "f1",
            "bleu",
            "rouge",
            "embedding_similarity",
            "bertscore",
            "llm_judge",
            "custom",
        }
        if self.metric not in valid_metrics:
            raise ValueError(
                f"Invalid metric: {self.metric}. Must be one of {valid_metrics}"
            )

        if self.metric == "custom" and self.custom_metric_path is None:
            raise ValueError(
                "custom_metric_path must be specified when using custom metric"
            )

        if self.max_training_samples is not None and self.max_training_samples <= 0:
            raise ValueError(
                f"max_training_samples must be positive, "
                f"got {self.max_training_samples}"
            )

        if self.max_validation_samples is not None and self.max_validation_samples <= 0:
            raise ValueError(
                f"max_validation_samples must be positive, "
                f"got {self.max_validation_samples}"
            )
