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
from typing import Optional

from oumi.core.configs.base_config import BaseConfig
from oumi.core.configs.inference_engine_type import InferenceEngineType
from oumi.core.configs.params.evaluation_params import EvaluationTaskParams
from oumi.core.configs.params.generation_params import GenerationParams
from oumi.core.configs.params.model_params import ModelParams
from oumi.core.configs.params.remote_params import RemoteParams
from oumi.utils.str_utils import sanitize_run_name


@dataclass
class EvaluationConfig(BaseConfig):
    tasks: list[EvaluationTaskParams] = field(default_factory=list)
    """List of all the evaluation tasks to run."""

    model: ModelParams = field(default_factory=ModelParams)
    """Parameters for the model to be evaluated.

    This includes model architecture, size, dtype,
    and any specific configurations required for the evaluation task.
    """

    generation: GenerationParams = field(default_factory=GenerationParams)
    """Parameters for text generation during evaluation.

    This includes settings such as temperature, top-k, top-p,
    maximum length, and any other parameters that control the
    text generation process.
    """

    inference_engine: InferenceEngineType = InferenceEngineType.NATIVE
    """For evaluation tasks that require an inference step, such as AlpacaEval tasks, an
    inference engine is required to generate model responses. This parameter specifies
    the inference engine to use for generation. If not defined, the default is the
    `NATIVE` inference engine."""

    inference_remote_params: Optional[RemoteParams] = None
    """For evaluation tasks that require an inference step, such as AlpacaEval tasks, an
    inference engine is required to generate model responses. If the model is accessed
    via a remote API, these parameters specify how to run inference against the remote
    API."""

    run_name: Optional[str] = None
    """A unique identifier for the current training run.
    This name is used to identify the run in Weights & Biases.
    """

    enable_wandb: bool = False
    """Whether to enable Weights & Biases (wandb) logging.
    Currently, this is only supported for LM Harness evaluation.
    If True, wandb will be used for experiment tracking and visualization.
    After enabling, you must set the `WANDB_API_KEY` environment variable.
    Alternatively, you can use the `wandb login` command to authenticate.
    """

    output_dir: str = "output"
    """Where to write computed evaluations."""

    def __post_init__(self):
        """Verifies params."""
        self.run_name = sanitize_run_name(self.run_name)
