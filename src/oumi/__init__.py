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
    - :func:`~oumi.chat`: Simple one-line chat interface for model inference.
    - :func:`~oumi.train.train`: Train a machine learning model.
    - :func:`~oumi.evaluate_async.evaluate_async`: Asynchronously evaluate a model.
    - :func:`~oumi.evaluate.evaluate`: Evaluate a model using LM Harness.
    - :func:`~oumi.infer.infer`: Perform inference with a trained model.
    - :func:`~oumi.infer.infer_interactive`: Run interactive inference with a model.
    - :func:`~oumi.quantize.quantize`: Quantize a model to reduce size and memory usage.
    - :func:`~oumi.judge.judge`: Judge a dataset using an LLM-as-judge approach.
    - :func:`~oumi.judge.judge_dataset`: Judge a dataset using a model.

Examples:
    Simple chat (one-liner)::

        >>> from oumi import chat
        >>> response = chat("gpt-4o", "What is machine learning?")

    Simple training::

        >>> from oumi import train
        >>> train("meta-llama/Llama-3.1-8B", "tatsu-lab/alpaca")

    Simple evaluation::

        >>> from oumi import evaluate
        >>> results = evaluate("meta-llama/Llama-3.1-8B", tasks=["mmlu"])

    Simple judging::

        >>> from oumi import judge
        >>> results = judge("gpt-4o", dataset, criteria="truthfulness")

    Training with full config::

        >>> from oumi import train
        >>> from oumi.core.configs import TrainingConfig
        >>> config = TrainingConfig(...)
        >>> train(config)

    Evaluating with full config::

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

    Judging with full config::

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
        JudgeConfig,
        QuantizationConfig,
        SynthesisConfig,
        TrainingConfig,
        TuningConfig,
    )
    from oumi.core.inference import BaseInferenceEngine
    from oumi.core.types.conversation import Conversation
    from oumi.judges.base_judge import JudgeOutput
    from oumi.quantize.base import QuantizationResult

logging.configure_dependency_warnings()


def chat(
    model: str,
    message: str | None = None,
    *,
    messages: list[dict[str, str]] | None = None,
    system_prompt: str | None = None,
    conversation: Conversation | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    top_p: float | None = None,
    return_conversation: bool = False,
    use_cache: bool = True,
) -> str | Conversation:
    """Simple one-line chat interface for model inference.

    This function provides a streamlined way to interact with various LLM providers
    with automatic provider detection based on model name.

    Args:
        model: Model name with optional provider prefix.
            - "gpt-4o" -> auto-detected as OpenAI
            - "claude-3-opus" -> auto-detected as Anthropic
            - "openai/gpt-4o" -> explicit OpenAI
            - "meta-llama/Llama-3.1-8B-Instruct" -> HuggingFace model via vLLM
            - "config.yaml" -> load full InferenceConfig from YAML
        message: The user message to send. Required unless using messages/conversation.
        messages: List of message dicts with 'role' and 'content' keys (OpenAI format).
            Alternative to using message + system_prompt for multi-turn conversations.
        system_prompt: Optional system prompt for the conversation.
        conversation: Optional existing Conversation object to continue.
        temperature: Sampling temperature (default: provider-specific).
        max_tokens: Maximum tokens to generate (default: 1024).
        top_p: Top-p sampling parameter.
        return_conversation: If True, return full Conversation object instead of string.
        use_cache: If True, cache and reuse inference engines for better performance.

    Returns:
        str: The assistant's response text (default).
        Conversation: Full conversation object if return_conversation=True.

    Examples:
        Simple single message::

            >>> response = chat("gpt-4o", "What is machine learning?")

        With parameters::

            >>> response = chat("claude-3-opus", "Explain AI", temperature=0.7)

        Multi-turn with dict messages (OpenAI format)::

            >>> response = chat("gpt-4o", messages=[
            ...     {"role": "system", "content": "You are helpful."},
            ...     {"role": "user", "content": "Hello!"},
            ... ])

        Continue a conversation::

            >>> conv = chat("gpt-4o", "Hi!", return_conversation=True)
            >>> response = chat("gpt-4o", "Tell me more", conversation=conv)
    """
    import oumi.infer

    return oumi.infer.chat(
        model,
        message,
        messages=messages,
        system_prompt=system_prompt,
        conversation=conversation,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        return_conversation=return_conversation,
        use_cache=use_cache,
    )


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


def evaluate(
    config_or_model: EvaluationConfig | str,
    tasks: list[str] | None = None,
    *,
    num_samples: int | None = None,
    batch_size: int = 8,
    output_dir: str = "./eval_output",
    enable_wandb: bool = False,
) -> list[dict[str, Any]]:
    """Evaluate a model using LM Harness or custom evaluation tasks.

    This function supports two modes:

    1. **Config mode**: Pass an EvaluationConfig object or YAML path
        >>> evaluate(EvaluationConfig.from_yaml("eval.yaml"))
        >>> evaluate("eval_config.yaml")

    2. **Simple mode**: Pass model name and task list
        >>> evaluate("meta-llama/Llama-3.1-8B", tasks=["mmlu", "hellaswag"])

    Args:
        config_or_model: Either an EvaluationConfig object, a YAML config path,
            or a model name (HuggingFace model ID or provider-prefixed name).
        tasks: List of evaluation task names (e.g., ["mmlu", "hellaswag"]).
            Only used in simple mode. See LM Harness for available tasks.
        num_samples: Number of samples per task (None = all samples).
            Only used in simple mode.
        batch_size: Batch size for evaluation. Only used in simple mode.
        output_dir: Output directory for evaluation results. Only used in simple mode.
        enable_wandb: Whether to enable W&B logging. Only used in simple mode.

    Returns:
        A list of evaluation results (one for each task). Each evaluation result is a
        dictionary of metric names and their corresponding values.

    Examples:
        Simple evaluation with default settings::

            >>> results = evaluate("meta-llama/Llama-3.1-8B", tasks=["mmlu"])

        Evaluation with sampling::

            >>> results = evaluate(
            ...     "gpt-4o",
            ...     tasks=["mmlu", "hellaswag"],
            ...     num_samples=100,
            ... )

        From YAML config::

            >>> results = evaluate("eval_config.yaml")

        Full config object::

            >>> results = evaluate(EvaluationConfig(...))
    """
    import oumi.evaluate

    return oumi.evaluate.evaluate(
        config_or_model,
        tasks,
        num_samples=num_samples,
        batch_size=batch_size,
        output_dir=output_dir,
        enable_wandb=enable_wandb,
    )


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
    inference_engine: BaseInferenceEngine | None = None,
    *,
    input_image_bytes: list[bytes] | None = None,
) -> list[Conversation]:
    """Runs batch inference for a model using the provided configuration.

    Args:
        config: The configuration to use for inference.
        inputs: A list of inputs for inference.
        inference_engine: The engine to use for inference. If unspecified, the engine
            will be inferred from `config`.
        input_image_bytes: A list of input PNG image bytes to be used with `image+text`
            VLMs. Only used in interactive mode.

    Returns:
        object: A list of model responses.
    """
    import oumi.infer

    return oumi.infer.infer(
        config, inputs, inference_engine, input_image_bytes=input_image_bytes
    )


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


def judge(
    config_or_model: JudgeConfig | str,
    dataset: list[dict[str, str]],
    *,
    criteria: str | None = None,
    prompt_template: str | None = None,
    judgment_type: str = "bool",
    include_explanation: bool = True,
    output_file: str | None = None,
) -> list[JudgeOutput]:
    """Judge a dataset using an LLM-as-judge approach.

    This function supports two modes:

    1. **Config mode**: Pass a JudgeConfig object or YAML path
        >>> judge(JudgeConfig(...), dataset)
        >>> judge("judge_config.yaml", dataset)

    2. **Simple mode**: Pass model name with criteria or custom prompt
        >>> judge("gpt-4o", dataset, criteria="truthfulness")
        >>> judge("gpt-4o", dataset, prompt_template="Is this accurate? {request} {response}")

    Args:
        config_or_model: Either a JudgeConfig object, a YAML config path,
            or a judge model name (HuggingFace model ID or provider-prefixed name).
        dataset: List of dictionaries containing input data. Each dictionary should
            have 'request' and 'response' keys (or keys matching your prompt placeholders).
        criteria: Predefined criteria name for simple mode. Available criteria:
            "truthfulness", "helpfulness", "safety", "relevance", "coherence".
        prompt_template: Custom prompt template with {request}, {response} placeholders.
            Use this instead of criteria for custom evaluation logic.
        judgment_type: Type of judgment output - "bool", "int", "float", "text", "enum".
            Only used in simple mode with custom prompt_template.
        include_explanation: Whether to request explanations from the judge.
            Only used in simple mode.
        output_file: Optional path to save results as JSONL.

    Returns:
        List of JudgeOutput objects containing judgment results.

    Examples:
        Simple judging with predefined criteria::

            >>> results = judge("gpt-4o", dataset, criteria="truthfulness")

        Custom prompt template::

            >>> results = judge(
            ...     "claude-3-opus",
            ...     dataset,
            ...     prompt_template="Is this response accurate? Q: {request} A: {response}",
            ...     judgment_type="bool",
            ... )

        From config file::

            >>> results = judge("judge_config.yaml", dataset)
    """
    import oumi.judge

    return oumi.judge.judge(
        config_or_model,
        dataset,
        criteria=criteria,
        prompt_template=prompt_template,
        judgment_type=judgment_type,
        include_explanation=include_explanation,
        output_file=output_file,
    )


def synthesize(config: SynthesisConfig) -> list[dict[str, Any]]:
    """Synthesize a dataset using the provided configuration."""
    import oumi.synth

    return oumi.synth.synthesize(config)


def train(
    config_or_model: TrainingConfig | str,
    dataset_or_model_kwargs: str | dict[str, Any] | None = None,
    additional_trainer_kwargs: dict[str, Any] | None = None,
    additional_tuning_kwargs: dict[str, Any] | None = None,
    verbose: bool = False,
    *,
    method: str = "sft",
    output_dir: str = "./output",
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    use_peft: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
) -> dict[str, Any] | None:
    """Train a model using Oumi's training framework.

    This function supports two modes:

    1. **Config mode**: Pass a TrainingConfig object or YAML path
        >>> train(TrainingConfig(...))
        >>> train("training_config.yaml")

    2. **Simple mode**: Pass model and dataset names with optional parameters
        >>> train("meta-llama/Llama-3.1-8B", "tatsu-lab/alpaca")
        >>> train("meta-llama/Llama-3.1-8B", "my-dataset", method="dpo")

    Args:
        config_or_model: Either a TrainingConfig object, a YAML config path,
            or a model name (HuggingFace model ID).
        dataset_or_model_kwargs: Either a dataset name (simple mode) or
            additional_model_kwargs dict (config mode).
        additional_trainer_kwargs: Additional kwargs to pass to the trainer.
        additional_tuning_kwargs: Additional kwargs to pass for hyperparameter tuning.
        verbose: Whether to print verbose output.
        method: Training method - "sft", "dpo", "kto", "grpo", "gkd", "gold".
            Only used in simple mode.
        output_dir: Output directory for checkpoints. Only used in simple mode.
        epochs: Number of training epochs. Only used in simple mode.
        batch_size: Per-device batch size. Only used in simple mode.
        learning_rate: Learning rate. Only used in simple mode.
        use_peft: Whether to use LoRA/PEFT. Only used in simple mode.
        lora_r: LoRA rank. Only used in simple mode.
        lora_alpha: LoRA alpha. Only used in simple mode.

    Returns:
        Training metrics dictionary, or None.

    Examples:
        Simple SFT training::

            >>> train("meta-llama/Llama-3.1-8B", "tatsu-lab/alpaca")

        DPO training with custom settings::

            >>> train(
            ...     "meta-llama/Llama-3.1-8B",
            ...     "my-preference-dataset",
            ...     method="dpo",
            ...     learning_rate=1e-5,
            ... )

        From YAML config::

            >>> train("training_config.yaml")

        Full config object::

            >>> train(TrainingConfig(...))
    """
    import oumi.train

    return oumi.train.train(
        config_or_model,
        dataset_or_model_kwargs,
        additional_trainer_kwargs=additional_trainer_kwargs,
        additional_tuning_kwargs=additional_tuning_kwargs,
        verbose=verbose,
        method=method,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        use_peft=use_peft,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
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


__all__ = [
    "chat",
    "evaluate_async",
    "evaluate",
    "infer_interactive",
    "infer",
    "judge",
    "judge_dataset",
    "quantize",
    "synthesize",
    "train",
    "tune",
]
