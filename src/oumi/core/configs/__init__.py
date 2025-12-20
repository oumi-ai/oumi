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

    This module uses lazy loading to improve import performance. Heavy dependencies
    like `peft` and `transformers` are only loaded when the corresponding config
    classes are accessed.
"""

from __future__ import annotations

import lazy_loader as lazy

# Use lazy.attach() to defer imports until they are actually accessed.
# This significantly improves import time by not loading heavy dependencies
# (like peft, transformers, trl) until they are needed.
_lazy_getattr, _lazy_dir, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        # Analyze config (lightweight)
        "analyze_config": [
            "AnalyzeConfig",
            "DatasetSource",
            "SampleAnalyzerParams",
        ],
        # Async evaluation config
        "async_evaluation_config": ["AsyncEvaluationConfig"],
        # Base config (lightweight - no heavy deps)
        "base_config": ["BaseConfig"],
        # Evaluation config
        "evaluation_config": ["EvaluationConfig"],
        # Inference config
        "inference_config": ["InferenceConfig"],
        # Inference engine type (lightweight enum)
        "inference_engine_type": ["InferenceEngineType"],
        # Job config (lightweight)
        "job_config": ["JobConfig", "JobResources", "StorageMount"],
        # Judge config
        "judge_config": ["JudgeConfig"],
        # Data params (lightweight)
        "params.data_params": [
            "DataParams",
            "DatasetParams",
            "DatasetSplit",
            "DatasetSplitParams",
            "MixtureStrategy",
        ],
        # Evaluation params (lightweight)
        "params.evaluation_params": [
            "EvaluationBackend",
            "EvaluationTaskParams",
            "LMHarnessTaskParams",
        ],
        # FSDP params (lightweight)
        "params.fsdp_params": [
            "AutoWrapPolicy",
            "BackwardPrefetch",
            "FSDPParams",
            "ShardingStrategy",
            "StateDictType",
        ],
        # Generation params (lightweight)
        "params.generation_params": ["GenerationParams"],
        # GRPO params (lightweight)
        "params.grpo_params": ["GrpoParams"],
        # Guided decoding params (lightweight)
        "params.guided_decoding_params": ["GuidedDecodingParams"],
        # Judge params (lightweight)
        "params.judge_params": [
            "JudgeOutputType",
            "JudgeResponseFormat",
        ],
        # Model params (imports transformers - HEAVY)
        "params.model_params": ["ModelParams"],
        # PEFT params (imports peft, transformers - HEAVY)
        "params.peft_params": [
            "LoraWeightInitialization",
            "PeftParams",
            "PeftSaveMode",
        ],
        # Profiler params (lightweight)
        "params.profiler_params": ["ProfilerParams"],
        # Remote params (lightweight)
        "params.remote_params": ["RemoteParams"],
        # Synthesis params (lightweight)
        # Note: DatasetSource from synthesis_params is exported as DatasetSourceParam
        # to avoid conflict with DatasetSource from analyze_config
        "params.synthesis_params": [
            "AttributeCombination",
            "DocumentSegmentationParams",
            "DocumentSource",
            "ExampleSource",
            "GeneralSynthesisParams",
            "GeneratedAttribute",
            "GeneratedAttributePostprocessingParams",
            "SampledAttribute",
            "SampledAttributeValue",
            "SegmentationStrategy",
            "TextConversation",
            "TextMessage",
            "TransformationStrategy",
            "TransformationType",
            "TransformedAttribute",
        ],
        # Telemetry params (lightweight)
        "params.telemetry_params": ["TelemetryParams"],
        # Training params (imports transformers, trl - HEAVY)
        "params.training_params": [
            "MixedPrecisionDtype",
            "SchedulerType",
            "TrainerType",
            "TrainingParams",
        ],
        # Tuning params (lightweight)
        "params.tuning_params": [
            "TunerType",
            "TuningParams",
        ],
        # Quantization config (lightweight)
        "quantization_config": ["QuantizationConfig"],
        # Synthesis config (lightweight)
        "synthesis_config": ["SynthesisConfig"],
        # Training config (imports training_params - HEAVY)
        "training_config": ["TrainingConfig"],
        # Tuning config (lightweight)
        "tuning_config": ["TuningConfig"],
    },
)

# Add DatasetSourceParam to __all__ (it's an alias for DatasetSource from synthesis_params)
__all__ = list(__all__) + ["DatasetSourceParam"]


def __getattr__(name: str):
    """Custom getattr to handle DatasetSourceParam alias."""
    if name == "DatasetSourceParam":
        # DatasetSourceParam is an alias for DatasetSource from synthesis_params
        from oumi.core.configs.params.synthesis_params import DatasetSource

        return DatasetSource
    return _lazy_getattr(name)


def __dir__():
    """Custom dir to include DatasetSourceParam."""
    return _lazy_dir()
