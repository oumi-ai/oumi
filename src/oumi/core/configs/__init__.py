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
    - :class:`~oumi.core.configs.params.judge_params.JudgeOutputType`
    - :class:`~oumi.core.configs.params.judge_params.JudgeResponseFormat`

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

    This module uses lazy imports to reduce startup time. Heavy dependencies like
    transformers, peft, and trl are only loaded when their associated config
    classes are actually accessed.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from oumi.exceptions import OumiConfigError as OumiConfigError

# Mapping of attribute names to their module paths
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # analyze_config
    "AnalyzeConfig": ("oumi.core.configs.analyze_config", "AnalyzeConfig"),
    "DatasetSource": ("oumi.core.configs.analyze_config", "DatasetSource"),
    "SampleAnalyzerParams": (
        "oumi.core.configs.analyze_config",
        "SampleAnalyzerParams",
    ),
    # async_evaluation_config
    "AsyncEvaluationConfig": (
        "oumi.core.configs.async_evaluation_config",
        "AsyncEvaluationConfig",
    ),
    # base_config
    "BaseConfig": ("oumi.core.configs.base_config", "BaseConfig"),
    # environment_config
    "EnvironmentConfig": (
        "oumi.core.configs.environment_config",
        "EnvironmentConfig",
    ),
    # evaluation_config
    "EvaluationConfig": ("oumi.core.configs.evaluation_config", "EvaluationConfig"),
    # inference_config
    "InferenceConfig": ("oumi.core.configs.inference_config", "InferenceConfig"),
    # inference_engine_type
    "InferenceEngineType": (
        "oumi.core.configs.inference_engine_type",
        "InferenceEngineType",
    ),
    # job_config
    "JobConfig": ("oumi.core.configs.job_config", "JobConfig"),
    "JobResources": ("oumi.core.configs.job_config", "JobResources"),
    "StorageMount": ("oumi.core.configs.job_config", "StorageMount"),
    # judge_config
    "JudgeConfig": ("oumi.core.configs.judge_config", "JudgeConfig"),
    # quantization_config
    "QuantizationConfig": (
        "oumi.core.configs.quantization_config",
        "QuantizationConfig",
    ),
    # synthesis_config
    "SynthesisConfig": ("oumi.core.configs.synthesis_config", "SynthesisConfig"),
    # training_config
    "TrainingConfig": ("oumi.core.configs.training_config", "TrainingConfig"),
    # tuning_config
    "TuningConfig": ("oumi.core.configs.tuning_config", "TuningConfig"),
    # params/data_params
    "DataParams": ("oumi.core.configs.params.data_params", "DataParams"),
    "DatasetParams": ("oumi.core.configs.params.data_params", "DatasetParams"),
    "DatasetSplit": ("oumi.core.configs.params.data_params", "DatasetSplit"),
    "DatasetSplitParams": (
        "oumi.core.configs.params.data_params",
        "DatasetSplitParams",
    ),
    "MixtureStrategy": ("oumi.core.configs.params.data_params", "MixtureStrategy"),
    "TrainTarget": ("oumi.core.configs.params.data_params", "TrainTarget"),
    # params/evaluation_params
    "EvaluationBackend": (
        "oumi.core.configs.params.evaluation_params",
        "EvaluationBackend",
    ),
    "EvaluationTaskParams": (
        "oumi.core.configs.params.evaluation_params",
        "EvaluationTaskParams",
    ),
    "LMHarnessTaskParams": (
        "oumi.core.configs.params.evaluation_params",
        "LMHarnessTaskParams",
    ),
    # params/fsdp_params
    "AutoWrapPolicy": ("oumi.core.configs.params.fsdp_params", "AutoWrapPolicy"),
    "BackwardPrefetch": ("oumi.core.configs.params.fsdp_params", "BackwardPrefetch"),
    "FSDPParams": ("oumi.core.configs.params.fsdp_params", "FSDPParams"),
    "ShardingStrategy": ("oumi.core.configs.params.fsdp_params", "ShardingStrategy"),
    "StateDictType": ("oumi.core.configs.params.fsdp_params", "StateDictType"),
    # params/generation_params
    "GenerationParams": (
        "oumi.core.configs.params.generation_params",
        "GenerationParams",
    ),
    # params/grpo_params
    "GrpoParams": ("oumi.core.configs.params.grpo_params", "GrpoParams"),
    # params/guided_decoding_params
    "GuidedDecodingParams": (
        "oumi.core.configs.params.guided_decoding_params",
        "GuidedDecodingParams",
    ),
    # params/judge_params
    "JudgeOutputType": ("oumi.core.configs.params.judge_params", "JudgeOutputType"),
    "JudgeResponseFormat": (
        "oumi.core.configs.params.judge_params",
        "JudgeResponseFormat",
    ),
    # params/model_params
    "ModelParams": ("oumi.core.configs.params.model_params", "ModelParams"),
    # params/peft_params
    "LoraWeightInitialization": (
        "oumi.core.configs.params.peft_params",
        "LoraWeightInitialization",
    ),
    "PeftParams": ("oumi.core.configs.params.peft_params", "PeftParams"),
    "PeftSaveMode": ("oumi.core.configs.params.peft_params", "PeftSaveMode"),
    # params/profiler_params
    "ProfilerParams": ("oumi.core.configs.params.profiler_params", "ProfilerParams"),
    # params/remote_params
    "RemoteParams": ("oumi.core.configs.params.remote_params", "RemoteParams"),
    # params/synthesis_params
    "AttributeCombination": (
        "oumi.core.configs.params.synthesis_params",
        "AttributeCombination",
    ),
    "DatasetSourceParam": (
        "oumi.core.configs.params.synthesis_params",
        "DatasetSource",
    ),
    "DocumentSegmentationParams": (
        "oumi.core.configs.params.synthesis_params",
        "DocumentSegmentationParams",
    ),
    "DocumentSource": ("oumi.core.configs.params.synthesis_params", "DocumentSource"),
    "ExampleSource": ("oumi.core.configs.params.synthesis_params", "ExampleSource"),
    "GeneralSynthesisParams": (
        "oumi.core.configs.params.synthesis_params",
        "GeneralSynthesisParams",
    ),
    "GeneratedAttribute": (
        "oumi.core.configs.params.synthesis_params",
        "GeneratedAttribute",
    ),
    "GeneratedAttributePostprocessingParams": (
        "oumi.core.configs.params.synthesis_params",
        "GeneratedAttributePostprocessingParams",
    ),
    "MultiTurnAttribute": (
        "oumi.core.configs.params.synthesis_params",
        "MultiTurnAttribute",
    ),
    "SampledAttribute": (
        "oumi.core.configs.params.synthesis_params",
        "SampledAttribute",
    ),
    "SampledAttributeValue": (
        "oumi.core.configs.params.synthesis_params",
        "SampledAttributeValue",
    ),
    "SegmentationStrategy": (
        "oumi.core.configs.params.synthesis_params",
        "SegmentationStrategy",
    ),
    "TextConversation": (
        "oumi.core.configs.params.synthesis_params",
        "TextConversation",
    ),
    "TextMessage": ("oumi.core.configs.params.synthesis_params", "TextMessage"),
    "TransformationStrategy": (
        "oumi.core.configs.params.synthesis_params",
        "TransformationStrategy",
    ),
    "TransformationType": (
        "oumi.core.configs.params.synthesis_params",
        "TransformationType",
    ),
    "TransformedAttribute": (
        "oumi.core.configs.params.synthesis_params",
        "TransformedAttribute",
    ),
    # params/telemetry_params
    "TelemetryParams": ("oumi.core.configs.params.telemetry_params", "TelemetryParams"),
    # params/training_params
    "MixedPrecisionDtype": (
        "oumi.core.configs.params.training_params",
        "MixedPrecisionDtype",
    ),
    "SchedulerType": ("oumi.core.configs.params.training_params", "SchedulerType"),
    "TrainerType": ("oumi.core.configs.params.training_params", "TrainerType"),
    "TrainingParams": ("oumi.core.configs.params.training_params", "TrainingParams"),
    # params/tuning_params
    "TunerType": ("oumi.core.configs.params.tuning_params", "TunerType"),
    "TuningParams": ("oumi.core.configs.params.tuning_params", "TuningParams"),
}

# Cache for already-imported attributes
_CACHE: dict[str, Any] = {}


def __getattr__(name: str) -> Any:
    """Lazily import and return the requested attribute."""
    if name in _CACHE:
        return _CACHE[name]

    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path)
        attr = getattr(module, attr_name)
        _CACHE[name] = attr
        return attr

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return the list of available attributes for autocompletion."""
    return list(__all__)


# For type checking, we still want the types to be available
if TYPE_CHECKING:
    from oumi.core.configs.analyze_config import (
        AnalyzeConfig as AnalyzeConfig,
    )
    from oumi.core.configs.analyze_config import (
        DatasetSource as DatasetSource,
    )
    from oumi.core.configs.analyze_config import (
        SampleAnalyzerParams as SampleAnalyzerParams,
    )
    from oumi.core.configs.async_evaluation_config import (
        AsyncEvaluationConfig as AsyncEvaluationConfig,
    )
    from oumi.core.configs.base_config import BaseConfig as BaseConfig
    from oumi.core.configs.environment_config import (
        EnvironmentConfig as EnvironmentConfig,
    )
    from oumi.core.configs.evaluation_config import (
        EvaluationConfig as EvaluationConfig,
    )
    from oumi.core.configs.inference_config import InferenceConfig as InferenceConfig
    from oumi.core.configs.inference_engine_type import (
        InferenceEngineType as InferenceEngineType,
    )
    from oumi.core.configs.job_config import (
        JobConfig as JobConfig,
    )
    from oumi.core.configs.job_config import (
        JobResources as JobResources,
    )
    from oumi.core.configs.job_config import (
        StorageMount as StorageMount,
    )
    from oumi.core.configs.judge_config import JudgeConfig as JudgeConfig
    from oumi.core.configs.params.data_params import (
        DataParams as DataParams,
    )
    from oumi.core.configs.params.data_params import (
        DatasetParams as DatasetParams,
    )
    from oumi.core.configs.params.data_params import (
        DatasetSplit as DatasetSplit,
    )
    from oumi.core.configs.params.data_params import (
        DatasetSplitParams as DatasetSplitParams,
    )
    from oumi.core.configs.params.data_params import (
        MixtureStrategy as MixtureStrategy,
    )
    from oumi.core.configs.params.data_params import (
        TrainTarget as TrainTarget,
    )
    from oumi.core.configs.params.evaluation_params import (
        EvaluationBackend as EvaluationBackend,
    )
    from oumi.core.configs.params.evaluation_params import (
        EvaluationTaskParams as EvaluationTaskParams,
    )
    from oumi.core.configs.params.evaluation_params import (
        LMHarnessTaskParams as LMHarnessTaskParams,
    )
    from oumi.core.configs.params.fsdp_params import (
        AutoWrapPolicy as AutoWrapPolicy,
    )
    from oumi.core.configs.params.fsdp_params import (
        BackwardPrefetch as BackwardPrefetch,
    )
    from oumi.core.configs.params.fsdp_params import (
        FSDPParams as FSDPParams,
    )
    from oumi.core.configs.params.fsdp_params import (
        ShardingStrategy as ShardingStrategy,
    )
    from oumi.core.configs.params.fsdp_params import (
        StateDictType as StateDictType,
    )
    from oumi.core.configs.params.generation_params import (
        GenerationParams as GenerationParams,
    )
    from oumi.core.configs.params.grpo_params import GrpoParams as GrpoParams
    from oumi.core.configs.params.guided_decoding_params import (
        GuidedDecodingParams as GuidedDecodingParams,
    )
    from oumi.core.configs.params.judge_params import (
        JudgeOutputType as JudgeOutputType,
    )
    from oumi.core.configs.params.judge_params import (
        JudgeResponseFormat as JudgeResponseFormat,
    )
    from oumi.core.configs.params.model_params import ModelParams as ModelParams
    from oumi.core.configs.params.peft_params import (
        LoraWeightInitialization as LoraWeightInitialization,
    )
    from oumi.core.configs.params.peft_params import (
        PeftParams as PeftParams,
    )
    from oumi.core.configs.params.peft_params import (
        PeftSaveMode as PeftSaveMode,
    )
    from oumi.core.configs.params.profiler_params import (
        ProfilerParams as ProfilerParams,
    )
    from oumi.core.configs.params.remote_params import RemoteParams as RemoteParams
    from oumi.core.configs.params.synthesis_params import (
        AttributeCombination as AttributeCombination,
    )
    from oumi.core.configs.params.synthesis_params import (
        DatasetSource as DatasetSourceParam,
    )
    from oumi.core.configs.params.synthesis_params import (
        DocumentSegmentationParams as DocumentSegmentationParams,
    )
    from oumi.core.configs.params.synthesis_params import (
        DocumentSource as DocumentSource,
    )
    from oumi.core.configs.params.synthesis_params import (
        ExampleSource as ExampleSource,
    )
    from oumi.core.configs.params.synthesis_params import (
        GeneralSynthesisParams as GeneralSynthesisParams,
    )
    from oumi.core.configs.params.synthesis_params import (
        GeneratedAttribute as GeneratedAttribute,
    )
    from oumi.core.configs.params.synthesis_params import (
        GeneratedAttributePostprocessingParams as GeneratedAttributePostprocessingParams,  # noqa: E501
    )
    from oumi.core.configs.params.synthesis_params import (
        MultiTurnAttribute as MultiTurnAttribute,
    )
    from oumi.core.configs.params.synthesis_params import (
        SampledAttribute as SampledAttribute,
    )
    from oumi.core.configs.params.synthesis_params import (
        SampledAttributeValue as SampledAttributeValue,
    )
    from oumi.core.configs.params.synthesis_params import (
        SegmentationStrategy as SegmentationStrategy,
    )
    from oumi.core.configs.params.synthesis_params import (
        TextConversation as TextConversation,
    )
    from oumi.core.configs.params.synthesis_params import (
        TextMessage as TextMessage,
    )
    from oumi.core.configs.params.synthesis_params import (
        TransformationStrategy as TransformationStrategy,
    )
    from oumi.core.configs.params.synthesis_params import (
        TransformationType as TransformationType,
    )
    from oumi.core.configs.params.synthesis_params import (
        TransformedAttribute as TransformedAttribute,
    )
    from oumi.core.configs.params.telemetry_params import (
        TelemetryParams as TelemetryParams,
    )
    from oumi.core.configs.params.training_params import (
        MixedPrecisionDtype as MixedPrecisionDtype,
    )
    from oumi.core.configs.params.training_params import (
        SchedulerType as SchedulerType,
    )
    from oumi.core.configs.params.training_params import (
        TrainerType as TrainerType,
    )
    from oumi.core.configs.params.training_params import (
        TrainingParams as TrainingParams,
    )
    from oumi.core.configs.params.tuning_params import (
        TunerType as TunerType,
    )
    from oumi.core.configs.params.tuning_params import (
        TuningParams as TuningParams,
    )
    from oumi.core.configs.quantization_config import (
        QuantizationConfig as QuantizationConfig,
    )
    from oumi.core.configs.synthesis_config import SynthesisConfig as SynthesisConfig
    from oumi.core.configs.training_config import TrainingConfig as TrainingConfig
    from oumi.core.configs.tuning_config import TuningConfig as TuningConfig

__all__ = [
    "AsyncEvaluationConfig",
    "AutoWrapPolicy",
    "BackwardPrefetch",
    "BaseConfig",
    "DataParams",
    "DatasetParams",
    "DatasetSplit",
    "DatasetSplitParams",
    "AnalyzeConfig",
    "DatasetSource",
    "SampleAnalyzerParams",
    "EvaluationTaskParams",
    "EvaluationConfig",
    "EvaluationBackend",
    "EvaluationConfig",
    "EnvironmentConfig",
    "EvaluationTaskParams",
    "FSDPParams",
    "GenerationParams",
    "GrpoParams",
    "GuidedDecodingParams",
    "InferenceConfig",
    "InferenceEngineType",
    "JobConfig",
    "JobResources",
    "JudgeConfig",
    "JudgeOutputType",
    "JudgeResponseFormat",
    "LMHarnessTaskParams",
    "LoraWeightInitialization",
    "MixedPrecisionDtype",
    "TrainTarget",
    "MixtureStrategy",
    "ModelParams",
    "OumiConfigError",
    "PeftParams",
    "PeftSaveMode",
    "ProfilerParams",
    "QuantizationConfig",
    "RemoteParams",
    "SchedulerType",
    "ShardingStrategy",
    "StateDictType",
    "StorageMount",
    "SynthesisConfig",
    "TelemetryParams",
    "TrainerType",
    "TrainingConfig",
    "TrainingParams",
    "TunerType",
    "TuningConfig",
    "TuningParams",
    "DatasetSourceParam",
    "DocumentSegmentationParams",
    "DocumentSource",
    "ExampleSource",
    "GeneratedAttributePostprocessingParams",
    "GeneralSynthesisParams",
    "GeneratedAttribute",
    "SampledAttribute",
    "SampledAttributeValue",
    "SegmentationStrategy",
    "TextConversation",
    "TextMessage",
    "TransformationStrategy",
    "TransformationType",
    "TransformedAttribute",
    "MultiTurnAttribute",
]
