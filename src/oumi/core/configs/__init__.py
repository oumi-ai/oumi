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

"""Configuration classes for Oumi."""

from oumi.core.configs.analyze_config import (
    DatasetAnalyzeConfig,
    SampleAnalyzeConfig,
)
from oumi.core.configs.async_evaluation_config import AsyncEvaluationConfig
from oumi.core.configs.base_config import BaseConfig
from oumi.core.configs.evaluation_config import (
    EvaluationConfig,
    EvaluationTaskParams,
)
from oumi.core.configs.inference_config import InferenceConfig
from oumi.core.configs.inference_engine_type import InferenceEngineType
from oumi.core.configs.job_config import JobConfig, JobResources, StorageMount
from oumi.core.configs.judge_config import (
    JudgeAttribute,
    JudgeAttributeValueType,
    JudgeConfig,
)
from oumi.core.configs.judge_config_v2 import (
    JudgeConfig as JudgeConfigV2,
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
    LMHarnessTaskParams,
)
from oumi.core.configs.params.evaluation_params import (
    EvaluationTaskParams as EvaluationTaskParamsV2,
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
from oumi.core.configs.params.judge_params import (
    JudgeOutputType,
    JudgeResponseFormat,
)
from oumi.core.configs.params.model_params import ModelParams
from oumi.core.configs.params.peft_params import (
    LoraWeightInitialization,
    PeftParams,
    PeftSaveMode,
)
from oumi.core.configs.params.profiler_params import ProfilerParams
from oumi.core.configs.params.remote_params import RemoteParams
from oumi.core.configs.params.synthesis_params import (
    DatasetSource,
)
from oumi.core.configs.params.telemetry_params import TelemetryParams
from oumi.core.configs.params.training_params import (
    MixedPrecisionDtype,
    SchedulerType,
    TrainerType,
    TrainingParams,
)
from oumi.core.configs.training_config import TrainingConfig

__all__ = [
    # Base config
    "BaseConfig",
    # Training configs
    "TrainingConfig",
    "TrainingParams",
    "TrainerType",
    "MixedPrecisionDtype",
    # Data configs
    "DataParams",
    "DatasetParams",
    "DatasetSplit",
    "DatasetSplitParams",
    "MixtureStrategy",
    # Model configs
    "ModelParams",
    # PEFT configs
    "PeftParams",
    "LoraWeightInitialization",
    "PeftSaveMode",
    # FSDP configs
    "FSDPParams",
    "AutoWrapPolicy",
    "BackwardPrefetch",
    "ShardingStrategy",
    "StateDictType",
    # Analysis configs
    "DatasetAnalyzeConfig",
    "SampleAnalyzeConfig",
    # Evaluation configs
    "EvaluationConfig",
    "EvaluationTaskParams",
    "AlpacaEvalTaskParams",
    "EvaluationBackend",
    "EvaluationTaskParamsV2",
    "LMHarnessTaskParams",
    # Inference configs
    "InferenceConfig",
    # Job configs
    "JobConfig",
    "JobResources",
    "StorageMount",
    # Judge configs
    "JudgeAttribute",
    "JudgeAttributeValueType",
    "JudgeConfig",
    "JudgeConfigV2",
    "JudgeOutputType",
    "JudgeResponseFormat",
    # Launcher config
    # Synthesis configs
    "DatasetSource",
    # Profiler, remote, telemetry, generation, grpo, guided decoding
    "ProfilerParams",
    "RemoteParams",
    "TelemetryParams",
    "GenerationParams",
    "GrpoParams",
    "GuidedDecodingParams",
    "SchedulerType",
    # Inference engine type
    "InferenceEngineType",
    # Async evaluation config
    "AsyncEvaluationConfig",
]
