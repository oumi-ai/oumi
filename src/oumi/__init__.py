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

"""Oumi (Open Universal Machine Intelligence) library.

This library provides tools and utilities for training, evaluating, and
inferring with machine learning models, particularly focused on language tasks.

Modules:
    - :mod:`~oumi.models`: Contains model architectures and related utilities.
    - :mod:`~oumi.evaluate`: Functions for evaluating models.
    - :mod:`~oumi.evaluate_async`: Asynchronous evaluation functionality.
    - :mod:`~oumi.infer`: Functions for model inference, including interactive mode.
    - :mod:`~oumi.train`: Training utilities for machine learning models.
    - :mod:`~oumi.utils`: Utility functions, including logging configuration.
    - :mod:`~oumi.judges`: Functions for judging datasets and conversations.

Functions:
    - :func:`~oumi.train.train`: Train a machine learning model.
    - :func:`~oumi.evaluate_async.evaluate_async`: Asynchronously evaluate a model.
    - :func:`~oumi.evaluate.evaluate`: Evaluate a model using LM Harness.
    - :func:`~oumi.infer.infer`: Perform inference with a trained model.
    - :func:`~oumi.infer.infer_interactive`: Run interactive inference with a model.
    - :func:`~oumi.quantize.quantize`: Quantize a model to reduce size and memory usage.
    - :func:`~oumi.judge.judge_dataset`: Judge a dataset using a model.

Examples:
    Training a model::

        >>> from oumi import train
        >>> from oumi.core.configs import TrainingConfig
        >>> config = TrainingConfig(...)
        >>> train(config)

    Evaluating a model::

        >>> from oumi import evaluate
        >>> from oumi.core.configs import EvaluationConfig
        >>> config = EvaluationConfig(...)
        >>> results = evaluate(config)

    Performing inference::

        >>> from oumi import infer
        >>> from oumi.core.configs import InferenceConfig
        >>> config = InferenceConfig(...)
        >>> outputs = infer(config)

    Quantizing a model::

        >>> from oumi import quantize
        >>> from oumi.core.configs import QuantizationConfig
        >>> config = QuantizationConfig(...)
        >>> result = quantize(config)

    Judging a dataset::
        >>> from oumi import judge_dataset
        >>> from oumi.core.configs import JudgeConfig
        >>> config = JudgeConfig(...)
        >>> judge_dataset(config, dataset)

    Tune a model::

        >>> from oumi import tune
        >>> from oumi.core.configs import TuningConfig
        >>> config = TuningConfig(...)
        >>> tune(config)

See Also:
    - :mod:`oumi.core.configs`: For configuration classes used in Oumi
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from oumi.utils import logging

if TYPE_CHECKING:
    from oumi.core.configs import (
        AsyncEvaluationConfig,
        EvaluationConfig,
        InferenceConfig,
        InferenceEngineType,
        JudgeConfig,
        QuantizationConfig,
        SynthesisConfig,
        TrainingConfig,
        TuningConfig,
    )
    from oumi.core.inference import BaseInferenceEngine
    from oumi.core.tokenizers import BaseTokenizer
    from oumi.core.types.conversation import Conversation
    from oumi.judges.base_judge import JudgeOutput
    from oumi.quantize.base import QuantizationResult

logging.configure_dependency_warnings()


def evaluate_async(config: AsyncEvaluationConfig) -> None:
    """Runs an async evaluation for a model using the provided configuration.

    Overview:
        This is a utility method for running evaluations iteratively over a series
        of checkpoints. This method can be run in parallel with a training job to
        compute metrics per checkpoint without wasting valuable time in the main
        training loop.

    Args:
        config: The desired configuration for evaluation.

    Returns:
        None.
    """
    import oumi.evaluate_async

    return oumi.evaluate_async.evaluate_async(config)


def evaluate(config: EvaluationConfig) -> list[dict[str, Any]]:
    """Evaluates a model using the provided configuration.

    Args:
        config: The desired configuration for evaluation.

    Returns:
        A list of evaluation results (one for each task). Each evaluation result is a
        dictionary of metric names and their corresponding values.
    """
    import oumi.evaluate

    return oumi.evaluate.evaluate(config)


def infer_interactive(
    config: InferenceConfig,
    *,
    input_image_bytes: list[bytes] | None = None,
    system_prompt: str | None = None,
) -> None:
    """Interactively provide the model response for a user-provided input."""
    import oumi.infer

    return oumi.infer.infer_interactive(
        config, input_image_bytes=input_image_bytes, system_prompt=system_prompt
    )


def infer(
    config: InferenceConfig,
    inputs: list[str] | None = None,
    *,
    inference_engine: BaseInferenceEngine | None = None,
    input_image_bytes: list[bytes] | None = None,
) -> list[Conversation]:
    """Run batch inference for a model using the provided configuration.

    Args:
        config: The configuration to use for inference.
        inputs: A list of text inputs (prompts) for inference.
        inference_engine: The engine to use for inference. If unspecified, the engine
            will be inferred from `config`.
        input_image_bytes: A list of input PNG image bytes to be used with `image+text`
            VLMs. Only used in interactive mode.

    Returns:
        A list of Conversation objects containing the model responses.

    Example:
        Using with InferenceConfig::

            from oumi import infer
            from oumi.core.configs import InferenceConfig

            config = InferenceConfig.for_model("gpt2", max_new_tokens=50)
            results = infer(config, inputs=["Hello, how are you?"])

    See Also:
        - :func:`quick_infer` for a simpler one-liner interface
        - :class:`InferenceConfig` for configuration options
    """
    import oumi.infer

    return oumi.infer.infer(
        config, inputs, inference_engine, input_image_bytes=input_image_bytes
    )


def quick_infer(
    model_name: str,
    inputs: list[str],
    *,
    max_new_tokens: int = 256,
    temperature: float | None = None,
    top_p: float | None = None,
    engine: InferenceEngineType | None = None,
    torch_dtype: str = "auto",
    device_map: str | None = "auto",
    trust_remote_code: bool = False,
    api_key: str | None = None,
    base_url: str | None = None,
) -> list[str]:
    """Run inference with minimal configuration - just model name and prompts.

    This is a convenience function for quickly running inference without
    constructing configuration objects. For more control, use :func:`infer`
    with an :class:`InferenceConfig`.

    Args:
        model_name: Name or path of the model (HuggingFace model ID, local path,
            or remote model name depending on engine).
        inputs: List of text prompts to generate responses for.
        max_new_tokens: Maximum number of tokens to generate per response.
        temperature: Sampling temperature (higher = more random). None uses default.
        top_p: Nucleus sampling probability threshold. None uses default.
        engine: The inference engine to use. If None, defaults to NATIVE for
            local models.
        torch_dtype: Data type for model parameters ("auto", "float16", "bfloat16").
        device_map: Device placement strategy ("auto", None, or specific device).
        trust_remote_code: Whether to trust remote code when loading the model.
        api_key: API key for remote inference engines (OpenAI, Anthropic, etc).
        base_url: Base URL for remote inference engines.

    Returns:
        List of generated text strings, one per input prompt.

    Example:
        Basic inference with a local model::

            from oumi import quick_infer

            responses = quick_infer(
                "HuggingFaceTB/SmolLM2-135M-Instruct",
                ["What is 2+2?", "What is the capital of France?"],
                max_new_tokens=100,
            )
            for response in responses:
                print(response)

        Using with OpenAI::

            from oumi import quick_infer
            from oumi.core.configs import InferenceEngineType

            responses = quick_infer(
                "gpt-4o-mini",
                ["Explain quantum computing in one sentence."],
                engine=InferenceEngineType.OPENAI,
                api_key="sk-...",
            )

    See Also:
        - :func:`infer` for full control with InferenceConfig
        - :class:`InferenceConfig.for_model` for reusable configuration
    """
    from oumi.core.configs import InferenceConfig

    config = InferenceConfig.for_model(
        model_name,
        engine=engine,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        api_key=api_key,
        base_url=base_url,
    )

    conversations = infer(config, inputs=inputs)

    # Extract the assistant's response text from each conversation
    results = []
    for conv in conversations:
        # Get the last assistant message
        assistant_msg = conv.last_message(role=None)
        if assistant_msg is not None:
            results.append(assistant_msg.compute_flattened_text_content())
        else:
            results.append("")

    return results


def judge_dataset(
    judge_config: JudgeConfig | str,
    dataset: list[dict[str, str]],
) -> list[JudgeOutput]:
    """Judge a dataset using Oumi's Judge framework.

    This function evaluates a dataset by instantiating a SimpleJudge with the provided
    configuration and running batch inference on all input data.

    The function performs the following steps:
        1. Initializes a SimpleJudge with the provided configuration.
        2. Passes the entire dataset to the judge for batch evaluation.
        3. Returns structured JudgeOutput objects containing parsed results.

    Args:
        judge_config: JudgeConfig object or path to a judge config file.
        dataset: List of dictionaries containing input data for evaluation. Each
            dictionary should contain key-value pairs that match placeholders in
            the judge's prompt template (e.g., {'question': '...', 'answer': '...'}).

    Returns:
        List[JudgeOutput]: A list of structured judgment results, each containing:
            - raw_output: The original response from the judge model
            - parsed_output: Extracted field values from structured formats (XML/JSON)
            - field_values: Typed values for each expected output field
            - field_scores: Numeric scores for applicable fields

    Example:
        >>> judge_config = JudgeConfig( # doctest: +SKIP
        ...     judge_params=JudgeParams(
        ...         prompt_template="Is this helpful? {question}, {answer}",
        ...         response_format=JudgeResponseFormat.XML,
        ...         judgment_type=JudgeOutputType.BOOL,
        ...         include_explanation=False
        ...     ),
        ...     inference_config=InferenceConfig(
        ...         model=ModelParams(model_name="gpt-4.1"),
        ...         generation=GenerationParams(max_tokens=100),
        ...         engine=InferenceEngineType.OPENAI
        ...     )
        ... )
        >>> dataset = [
        ...     {'question': 'What is 2+2?', 'answer': '4'},
        ...     {'question': 'How to cook?', 'answer': 'I dont know'}
        ... ]
        >>> judged_outputs = judge_dataset(judge_config, dataset)
        >>> for output in judged_outputs:
        ...     print(output.field_values)  # e.g., {'judgment': True}
    """
    import oumi.judge

    return oumi.judge.judge_dataset(judge_config, dataset)


def synthesize(config: SynthesisConfig) -> list[dict[str, Any]]:
    """Synthesize a dataset using the provided configuration."""
    import oumi.synth

    return oumi.synth.synthesize(config)


def train(
    config: TrainingConfig,
    *,
    additional_model_kwargs: dict[str, Any] | None = None,
    additional_trainer_kwargs: dict[str, Any] | None = None,
    verbose: bool = False,
) -> dict[str, Any] | None:
    """Train a model using the provided configuration.

    This is the main entry point for model training in Oumi. It handles
    distributed training setup, model/tokenizer loading, dataset preparation,
    and training loop execution.

    Args:
        config: Complete training configuration including model, data, and
            training parameters. See :class:`TrainingConfig` for details.
        additional_model_kwargs: Extra arguments passed to model constructor.
            Use for model-specific parameters not covered by ModelParams.
        additional_trainer_kwargs: Extra arguments passed to the trainer.
            Useful for accessing trainer-specific features.
        verbose: If True, logs detailed configuration information at startup.

    Returns:
        None for regular training. When used with hyperparameter tuning
        (via TuningConfig), returns a dict with evaluation metrics for
        optimization.

    Raises:
        ValueError: If required configuration fields are missing or invalid.
        RuntimeError: If training fails due to hardware or software issues.

    Example:
        Basic training from a YAML config file::

            from oumi import train
            from oumi.core.configs import TrainingConfig

            config = TrainingConfig.from_yaml("my_training_config.yaml")
            train(config)

        Training with programmatic configuration::

            from oumi import train
            from oumi.core.configs import (
                TrainingConfig,
                ModelParams,
                DataParams,
                TrainingParams,
                DatasetParams,
                DatasetSplitParams,
            )

            config = TrainingConfig(
                model=ModelParams(
                    model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
                    torch_dtype_str="bfloat16",
                ),
                data=DataParams(
                    train=DatasetSplitParams(
                        datasets=[DatasetParams(dataset_name="tatsu-lab/alpaca")]
                    )
                ),
                training=TrainingParams(
                    output_dir="./output",
                    num_train_epochs=3,
                    per_device_train_batch_size=8,
                ),
            )
            train(config, verbose=True)

    See Also:
        - :class:`oumi.core.configs.TrainingConfig` for configuration options
        - :func:`evaluate` for model evaluation
        - :func:`infer` for model inference
    """
    import oumi.train

    return oumi.train.train(
        config,
        additional_model_kwargs=additional_model_kwargs,
        additional_trainer_kwargs=additional_trainer_kwargs,
        verbose=verbose,
    )


def quantize(config: QuantizationConfig) -> QuantizationResult:
    """Quantizes a model using the provided configuration.

    Args:
        config: Quantization configuration containing model parameters,
            method, output path, and other settings.

    Returns:
        QuantizationResult containing:
        - quantized_size_bytes: Size of the quantized model in bytes
        - output_path: Path to the quantized model
        - quantization_method: Quantization method used
        - format_type: Format type of the quantized model
        - additional_info: Additional method-specific information

    Raises:
        RuntimeError: If quantization fails for any reason
        ValueError: If configuration is invalid for this quantizer
    """
    import oumi.quantize

    return oumi.quantize.quantize(config)


def tune(config: TuningConfig) -> None:
    """Tunes hyperparameters for a model using the provided configuration."""
    import oumi.tune

    return oumi.tune.tune(config)


def build_dataset(
    dataset_name: str,
    *,
    tokenizer: BaseTokenizer | None = None,
    seq_length: int | None = None,
    dataset_path: str | None = None,
    subset: str | None = None,
    split: str | None = None,
    trust_remote_code: bool = False,
    **kwargs: Any,
) -> Any:
    """Build a dataset by name from the Oumi registry or HuggingFace Hub.

    This is a convenience function for quickly loading datasets without
    constructing full configuration objects.

    Args:
        dataset_name: Name of the dataset in Oumi registry or HuggingFace Hub.
        tokenizer: Optional tokenizer for processing the dataset.
        seq_length: Optional maximum sequence length for tokenization.
        dataset_path: Optional path override for the dataset location.
        subset: Optional dataset subset/configuration name.
        split: Optional dataset split (e.g., "train", "validation", "test").
        trust_remote_code: Whether to trust remote code when loading datasets.
        **kwargs: Additional keyword arguments passed to the dataset constructor.

    Returns:
        The constructed dataset object.

    Example:
        >>> from oumi import build_dataset
        >>> dataset = build_dataset("tatsu-lab/alpaca", split="train")
    """
    import oumi.builders

    return oumi.builders.build_dataset(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        seq_length=seq_length,
        dataset_path=dataset_path,
        subset=subset,
        split=split,
        trust_remote_code=trust_remote_code,
        **kwargs,
    )


def build_model(
    model_name: str,
    *,
    tokenizer_name: str | None = None,
    torch_dtype: str = "auto",
    device_map: str | None = "auto",
    trust_remote_code: bool = False,
    **kwargs: Any,
) -> Any:
    """Build a model by name from HuggingFace Hub or local path.

    This is a convenience function for quickly loading models without
    constructing full configuration objects.

    Args:
        model_name: Name or path of the model (HuggingFace model ID or local path).
        tokenizer_name: Optional tokenizer name (defaults to model_name).
        torch_dtype: Data type for model parameters ("auto", "float16", "bfloat16", etc).
        device_map: Device placement strategy ("auto", None, or specific device).
        trust_remote_code: Whether to trust remote code when loading the model.
        **kwargs: Additional keyword arguments passed to the model constructor.

    Returns:
        The constructed model object.

    Example:
        >>> from oumi import build_model
        >>> model = build_model("meta-llama/Llama-3.2-1B", torch_dtype="bfloat16")
    """
    import oumi.builders
    from oumi.core.configs.params.model_params import ModelParams

    model_params = ModelParams(
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        torch_dtype_str=torch_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        **{k: v for k, v in kwargs.items() if k in ModelParams.__dataclass_fields__},
    )

    model_kwargs = {
        k: v for k, v in kwargs.items() if k not in ModelParams.__dataclass_fields__
    }

    return oumi.builders.build_model(model_params=model_params, **model_kwargs)


def build_tokenizer(
    model_name: str,
    *,
    tokenizer_name: str | None = None,
    trust_remote_code: bool = False,
    **kwargs: Any,
) -> BaseTokenizer:
    """Build a tokenizer by name from HuggingFace Hub or local path.

    This is a convenience function for quickly loading tokenizers without
    constructing full configuration objects.

    Args:
        model_name: Name or path of the model whose tokenizer to load.
        tokenizer_name: Optional explicit tokenizer name (defaults to model_name).
        trust_remote_code: Whether to trust remote code when loading the tokenizer.
        **kwargs: Additional keyword arguments passed to the tokenizer constructor.

    Returns:
        The constructed tokenizer object.

    Example:
        >>> from oumi import build_tokenizer
        >>> tokenizer = build_tokenizer("meta-llama/Llama-3.2-1B")
    """
    import oumi.builders
    from oumi.core.configs.params.model_params import ModelParams

    model_params = ModelParams(
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        trust_remote_code=trust_remote_code,
        tokenizer_kwargs=kwargs,
    )

    return oumi.builders.build_tokenizer(model_params)


__all__ = [
    # Core functions
    "evaluate_async",
    "evaluate",
    "infer_interactive",
    "infer",
    "quick_infer",
    "quantize",
    "synthesize",
    "train",
    "tune",
    # Builder convenience functions
    "build_dataset",
    "build_model",
    "build_tokenizer",
]
