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
    - :class:`~oumi.core.configs.inference_config.InferenceEngineType`
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
- Profiling:
    - :class:`~oumi.core.configs.params.profiler_params.ProfilerParams`
- Telemetry:
    - :class:`~oumi.core.configs.params.telemetry_params.TelemetryParams`
- Judge:
    - :class:`~oumi.core.configs.judge_config.JudgeConfig`

For more information on using these configurations, see the
:doc:`/get_started/configuration` guide.

Example:
    >>> from oumi.core.configs import TrainingConfig, ModelParams
    >>> model_params = ModelParams(model_name="gpt2", num_labels=2)
    >>> training_config = TrainingConfig(
    ...     model_params=model_params,
    ...     batch_size=32,
    ...     num_epochs=3
    ... )
    >>> # Use the training_config in your training pipeline

Note:
    All configuration classes inherit from
        :class:`~oumi.core.configs.base_config.BaseConfig`,
        which provides common functionality such as serialization and validation.
"""

from oumi.core.configs.async_evaluation_config import AsyncEvaluationConfig
from oumi.core.configs.base_config import BaseConfig
from oumi.core.configs.evaluation_config import (
    EvaluationConfig,
    EvaluationFramework,
)
from oumi.core.configs.inference_config import InferenceConfig, InferenceEngineType
from oumi.core.configs.job_config import JobConfig, JobResources, StorageMount
from oumi.core.configs.judge_config import JudgeAttribute, JudgeConfig
from oumi.core.configs.params.data_params import (
    DataParams,
    DatasetParams,
    DatasetSplit,
    DatasetSplitParams,
    MixtureStrategy,
)
from oumi.core.configs.params.evaluation_params import (
    CustomEvaluationParams,
    LMHarnessParams,
)
from oumi.core.configs.params.fsdp_params import (
    AutoWrapPolicy,
    BackwardPrefetch,
    FSDPParams,
    ShardingStrategy,
    StateDictType,
)
from oumi.core.configs.params.generation_params import GenerationParams
from oumi.core.configs.params.guided_decoding_params import GuidedDecodingParams
from oumi.core.configs.params.model_params import ModelParams
from oumi.core.configs.params.peft_params import PeftParams
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
    "AsyncEvaluationConfig",
    "AutoWrapPolicy",
    "BackwardPrefetch",
    "BaseConfig",
    "CustomEvaluationParams",
    "DataParams",
    "DatasetParams",
    "DatasetSplit",
    "DatasetSplitParams",
    "EvaluationConfig",
    "EvaluationFramework",
    "FSDPParams",
    "GenerationParams",
    "GuidedDecodingParams",
    "InferenceConfig",
    "InferenceEngineType",
    "JobConfig",
    "JobResources",
    "JudgeAttribute",
    "JudgeConfig",
    "LMHarnessParams",
    "MixedPrecisionDtype",
    "MixtureStrategy",
    "ModelParams",
    "PeftParams",
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
