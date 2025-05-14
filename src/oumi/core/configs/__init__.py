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

"""Configuration module for the Oumi (Open Universal Machine Intelligence) library.

This module provides various configuration classes and parameters used throughout
the Oumi framework for tasks such as training, evaluation, inference,
and job management.

The configurations are organized into different categories:

- Evaluation:
    - :class:`~oumi.core.configs.async_evaluation_config.AsyncEvaluationConfig`
    - :class:`~oumi.core.configs.evaluation_config.EvaluationConfig`
    - :class:`~oumi.core.configs.evaluation_config.EvaluationFramework`
- Generation and Inference:
    - :class:`~oumi.core.configs.params.generation_params.GenerationParams`
    - :class:`~oumi.core.configs.inference_config.InferenceConfig`
    - :class:`~oumi.core.configs.inference_engine_type.InferenceEngineType`
- Job Management:
    - :class:`~oumi.core.configs.job_config.JobConfig`
    - :class:`~oumi.core.configs.job_config.JobResources`
    - :class:`~oumi.core.configs.job_config.StorageMount`
- Data:
    - :class:`~oumi.core.configs.params.data_params.DataParams`
    - :class:`~oumi.core.configs.params.data_params.DatasetParams`
    - :class:`~oumi.core.configs.params.data_params.DatasetSplit`
    - :class:`~oumi.core.configs.params.data_params.DatasetSplitParams`
    - :class:`~oumi.core.configs.params.data_params.MixtureStrategy`
- Model:
    - :class:`~oumi.core.configs.params.model_params.ModelParams`
    - :class:`~oumi.core.configs.params.peft_params.PeftParams`
    - :class:`~oumi.core.configs.params.fsdp_params.FSDPParams`
- Training:
    - :class:`~oumi.core.configs.training_config.TrainingConfig`
    - :class:`~oumi.core.configs.params.training_params.TrainingParams`
    - :class:`~oumi.core.configs.params.training_params.MixedPrecisionDtype`
    - :class:`~oumi.core.configs.params.training_params.SchedulerType`
    - :class:`~oumi.core.configs.params.training_params.TrainerType`
    - :class:`~oumi.core.configs.params.peft_params.LoraWeightInitialization`
    - :class:`~oumi.core.configs.params.peft_params.PeftSaveMode`
    - :class:`~oumi.core.configs.params.grpo_params.GrpoParams`
- Profiling:
    - :class:`~oumi.core.configs.params.profiler_params.ProfilerParams`
- Telemetry:
    - :class:`~oumi.core.configs.params.telemetry_params.TelemetryParams`
- Judge:
    - :class:`~oumi.core.configs.judge_config.JudgeConfig`
    - :class:`~oumi.core.configs.judge_config.JudgeAttribute`
    - :class:`~oumi.core.configs.judge_config.JudgeAttributeValueType`

Example:
    >>> from oumi.core.configs import ModelParams, TrainingConfig, TrainingParams
    >>> model_params = ModelParams(model_name="gpt2")
    >>> training_params = TrainingParams(num_train_epochs=3)
    >>> training_config = TrainingConfig(
    ...     model=model_params,
    ...     training=training_params,
    ... )
    >>> # Use the training_config in your training pipeline

Note:
    All configuration classes inherit from
        :class:`~oumi.core.configs.base_config.BaseConfig`,
        which provides common functionality such as serialization and validation.
"""

from oumi.core.configs.async_evaluation_config import AsyncEvaluationConfig
from oumi.core.configs.base_config import BaseConfig
from oumi.core.configs.evaluation_config import EvaluationConfig
from oumi.core.configs.inference_config import InferenceConfig
from oumi.core.configs.inference_engine_type import InferenceEngineType
from oumi.core.configs.job_config import JobConfig, JobResources, StorageMount
from oumi.core.configs.judge_config import (
    JudgeAttribute,
    JudgeAttributeValueType,
    JudgeConfig,
)
from oumi.core.configs.params.data_params import (
    DataParams,
    DatasetParams,
    DatasetSplit,
    DatasetSplitParams,
    MixtureStrategy,
)
from oumi.core.configs.params.evaluation_params import (
    AlpacaEvalTaskParams,
    EvaluationBackend,
    EvaluationTaskParams,
    LMHarnessTaskParams,
)
from oumi.core.configs.params.fsdp_params import (
    AutoWrapPolicy,
    BackwardPrefetch,
    FSDPParams,
    ShardingStrategy,
    StateDictType,
)
from oumi.core.configs.params.generation_params import GenerationParams
from oumi.core.configs.params.grpo_params import GrpoParams
from oumi.core.configs.params.guided_decoding_params import GuidedDecodingParams
from oumi.core.configs.params.model_params import ModelParams
from oumi.core.configs.params.peft_params import (
    LoraWeightInitialization,
    PeftParams,
    PeftSaveMode,
)
from oumi.core.configs.params.profiler_params import ProfilerParams
from oumi.core.configs.params.remote_params import RemoteParams
from oumi.core.configs.params.telemetry_params import TelemetryParams
from oumi.core.configs.params.training_params import (
    MixedPrecisionDtype,
    SchedulerType,
    TrainerType,
    TrainingParams,
)
from oumi.core.configs.training_config import TrainingConfig

__all__ = [
    "AlpacaEvalTaskParams",
    "AsyncEvaluationConfig",
    "AutoWrapPolicy",
    "BackwardPrefetch",
    "BaseConfig",
    "DataParams",
    "DatasetParams",
    "DatasetSplit",
    "DatasetSplitParams",
    "EvaluationTaskParams",
    "EvaluationConfig",
    "EvaluationBackend",
    "FSDPParams",
    "GenerationParams",
    "GrpoParams",
    "GuidedDecodingParams",
    "InferenceConfig",
    "InferenceEngineType",
    "JobConfig",
    "JobResources",
    "JudgeAttribute",
    "JudgeAttributeValueType",
    "JudgeConfig",
    "LMHarnessTaskParams",
    "LoraWeightInitialization",
    "MixedPrecisionDtype",
    "MixtureStrategy",
    "ModelParams",
    "PeftParams",
    "PeftSaveMode",
    "ProfilerParams",
    "RemoteParams",
    "SchedulerType",
    "ShardingStrategy",
    "StateDictType",
    "StorageMount",
    "TelemetryParams",
    "TrainerType",
    "TrainingConfig",
    "TrainingParams",
]
