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
    - :mod:`~oumi.judges`: Functions for judging datasets and model responses.

Functions:
    - :func:`~oumi.train.train`: Train a machine learning model.
    - :func:`~oumi.evaluate_async.evaluate_async`: Asynchronously evaluate a model.
    - :func:`~oumi.evaluate.evaluate`: Evaluate a model using LM Harness.
    - :func:`~oumi.infer.infer`: Perform inference with a trained model.
    - :func:`~oumi.infer.infer_interactive`: Run interactive inference with a model.
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

    Judging a dataset::

        >>> from oumi import judge_dataset
        >>> from oumi.core.configs import JudgeConfig
        >>> config = JudgeConfig(...)
        >>> judge_dataset(config, dataset)

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
        TrainingConfig,
    )
    from oumi.core.datasets import BaseSftDataset
    from oumi.core.inference import BaseInferenceEngine
    from oumi.core.types.conversation import Conversation

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


def judge_conversations(
    config: JudgeConfig, judge_inputs: list[Conversation]
) -> list[dict[str, Any]]:
    """Judge a list of conversations.

    This function evaluates a list of conversations using the specified Judge.

    The function performs the following steps:

        1. Initializes the Judge with the provided configuration.
        2. Uses the Judge to evaluate each conversation input.
        3. Collects and returns the judged outputs.

    Args:
        config: The configuration for the judge.
        judge_inputs: A list of Conversation objects to be judged.

    Returns:
        List[Dict[str, Any]]: A list of judgement results for each conversation.

        >>> # Example output:
        [
            {'helpful': True, 'safe': False},
            {'helpful': True, 'safe': True},
        ]

    Example:
        >>> config = JudgeConfig(...) # doctest: +SKIP
        >>> judge_inputs = [Conversation(...), Conversation(...)] # doctest: +SKIP
        >>> judged_outputs = judge_conversations(config, judge_inputs) # doctest: +SKIP
        >>> for output in judged_outputs: # doctest: +SKIP
        ...     print(output)
    """
    import oumi.judge

    return oumi.judge.judge_conversations(config, judge_inputs)


def judge_dataset(config: JudgeConfig, dataset: BaseSftDataset) -> list[dict[str, Any]]:
    """Judge a dataset.

    This function evaluates a given dataset using a specified Judge configuration.

    The function performs the following steps:

        1. Initializes the Judge with the provided configuration.
        2. Iterates through the dataset to extract conversation inputs.
        3. Uses the Judge to evaluate each conversation input.
        4. Collects and returns the judged outputs.

    Args:
        config: The configuration for the judge.
        dataset: The dataset to be judged. This dataset
            should be compatible with the Supervised Finetuning Dataset class.

    Returns:
        List[Dict[str, Any]]: A list of judgement results for each conversation.

        >>> # Example output:
        [
            {'helpful': True, 'safe': False},
            {'helpful': True, 'safe': True},
        ]

    Example:
        >>> config = JudgeConfig(...) # doctest: +SKIP
        >>> dataset = SomeDataset(...) # doctest: +SKIP
        >>> judged_outputs = judge_dataset(config, dataset) # doctest: +SKIP
        >>> for output in judged_outputs: # doctest: +SKIP
        ...     print(output)
    """
    import oumi.judge

    return oumi.judge.judge_dataset(config, dataset)


def train(
    config: TrainingConfig,
    additional_model_kwargs: dict[str, Any] | None = None,
    additional_trainer_kwargs: dict[str, Any] | None = None,
) -> None:
    """Trains a model using the provided configuration."""
    import oumi.train

    return oumi.train.train(
        config,
        additional_model_kwargs=additional_model_kwargs,
        additional_trainer_kwargs=additional_trainer_kwargs,
    )


__all__ = [
    "evaluate_async",
    "evaluate",
    "infer_interactive",
    "infer",
    "judge_conversations",
    "judge_dataset",
    "train",
]
