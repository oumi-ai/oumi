"""Configuration module for the LeMa (Learning Machines) library.

This module provides various configuration classes and parameters used throughout
the LeMa framework for tasks such as training, evaluation, inference,
and job management.

The configurations are organized into different categories:
- Evaluation:
    - :class:`~lema.core.configs.async_evaluation_config.AsyncEvaluationConfig`
    - :class:`~lema.core.configs.evaluation_config.EvaluationConfig`
    - :class:`~lema.core.configs.evaluation_config.EvaluationFramework`
- Generation and Inference:
    - :class:`~lema.core.configs.generation_config.GenerationConfig`
    - :class:`~lema.core.configs.inference_config.InferenceConfig`
- Job Management:
    - :class:`~lema.core.configs.job_config.JobConfig`
    - :class:`~lema.core.configs.job_config.JobResources`
    - :class:`~lema.core.configs.job_config.StorageMount`
- Data:
    - :class:`~lema.core.configs.params.data_params.DataParams`
    - :class:`~lema.core.configs.params.data_params.DatasetParams`
    - :class:`~lema.core.configs.params.data_params.DatasetSplit`
    - :class:`~lema.core.configs.params.data_params.DatasetSplitParams`
    - :class:`~lema.core.configs.params.data_params.MixtureStrategy`
- Model:
    - :class:`~lema.core.configs.params.model_params.ModelParams`
    - :class:`~lema.core.configs.params.peft_params.PeftParams`
- Training:
    - :class:`~lema.core.configs.training_config.TrainingConfig`
    - :class:`~lema.core.configs.params.training_params.TrainingParams`
    - :class:`~lema.core.configs.params.training_params.MixedPrecisionDtype`
    - :class:`~lema.core.configs.params.training_params.SchedulerType`
    - :class:`~lema.core.configs.params.training_params.TrainerType`
- Profiling:
    - :class:`~lema.core.configs.params.profiler_params.ProfilerParams`

For more information on using these configurations, see the :ref:`configuration_guide`.

Example:
    >>> from lema.core.configs import TrainingConfig, ModelParams
    >>> model_params = ModelParams(model_name="gpt2", num_labels=2)
    >>> training_config = TrainingConfig(
    ...     model_params=model_params,
    ...     batch_size=32,
    ...     num_epochs=3
    ... )
    >>> # Use the training_config in your training pipeline

Note:
    All configuration classes inherit from
        :class:`~lema.core.configs.base_config.BaseConfig`,
        which provides common functionality such as serialization and validation.
"""

from lema.core.configs.async_evaluation_config import AsyncEvaluationConfig
from lema.core.configs.base_config import BaseConfig
from lema.core.configs.evaluation_config import (
    EvaluationConfig,
    EvaluationFramework,
)
from lema.core.configs.generation_config import GenerationConfig
from lema.core.configs.inference_config import InferenceConfig
from lema.core.configs.job_config import JobConfig, JobResources, StorageMount
from lema.core.configs.params.data_params import (
    DataParams,
    DatasetParams,
    DatasetSplit,
    DatasetSplitParams,
    MixtureStrategy,
)
from lema.core.configs.params.model_params import ModelParams
from lema.core.configs.params.peft_params import PeftParams
from lema.core.configs.params.profiler_params import ProfilerParams
from lema.core.configs.params.training_params import (
    MixedPrecisionDtype,
    SchedulerType,
    TrainerType,
    TrainingParams,
)
from lema.core.configs.training_config import TrainingConfig

__all__ = [
    "AsyncEvaluationConfig",
    "BaseConfig",
    "DataParams",
    "DatasetParams",
    "DatasetSplit",
    "DatasetSplitParams",
    "EvaluationConfig",
    "EvaluationFramework",
    "GenerationConfig",
    "InferenceConfig",
    "JobConfig",
    "JobResources",
    "MixtureStrategy",
    "MixedPrecisionDtype",
    "ModelParams",
    "PeftParams",
    "ProfilerParams",
    "SchedulerType",
    "StorageMount",
    "TrainerType",
    "TrainingConfig",
    "TrainingParams",
]
