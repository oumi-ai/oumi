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

from typing import Optional

from oumi.builders.inference_engines import build_inference_engine
from oumi.core.configs import InferenceConfig, InferenceEngineType
from oumi.core.inference import BaseInferenceEngine
from oumi.core.types.conversation import (
    ContentItem,
    Conversation,
    Message,
    Role,
    Type,
)
from oumi.utils.logging import logger


def _get_engine(config: InferenceConfig) -> BaseInferenceEngine:
    """Returns the inference engine based on the provided config."""
    if config.engine is None:
        logger.warning(
            "No inference engine specified. Using the default 'native' engine."
        )
    return build_inference_engine(
        engine_type=config.engine or InferenceEngineType.NATIVE,
        model_params=config.model,
        remote_params=config.remote_params,
    )


def infer_interactive(
    config: InferenceConfig,
    *,
    input_image_bytes: Optional[list[bytes]] = None,
    system_prompt: Optional[str] = None,
) -> None:
    """Interactively provide the model response for a user-provided input."""
    # Create engine up front to avoid reinitializing it for each input.
    inference_engine = _get_engine(config)
    while True:
        try:
            input_text = input("Enter your input prompt: ")
        except (EOFError, KeyboardInterrupt):  # Triggered by Ctrl+D/Ctrl+C
            print("\nExiting...")
            return
        model_response = infer(
            config=config,
            inputs=[
                input_text,
            ],
            system_prompt=system_prompt,
            input_image_bytes=input_image_bytes,
            inference_engine=inference_engine,
        )
        for g in model_response:
            print("------------")
            print(repr(g))
            print("------------")
        print()


def infer(
    config: InferenceConfig,
    inputs: Optional[list[str]] = None,
    inference_engine: Optional[BaseInferenceEngine] = None,
    *,
    input_image_bytes: Optional[list[bytes]] = None,
    system_prompt: Optional[str] = None,
) -> list[Conversation]:
    """Runs batch inference for a model using the provided configuration.

    Args:
        config: The configuration to use for inference.
        inputs: A list of inputs for inference.
        inference_engine: The engine to use for inference. If unspecified, the engine
            will be inferred from `config`.
        input_image_bytes: A list of input PNG image bytes to be used with `image+text`
            VLMs. Only used in interactive mode.
        system_prompt: System prompt for task-specific instructions.

    Returns:
        object: A list of model responses.
    """
    if not inference_engine:
        inference_engine = _get_engine(config)

    # Pass None if no conversations are provided.
    conversations = None
    if inputs is not None and len(inputs) > 0:
        system_messages = (
            [Message(role=Role.SYSTEM, content=system_prompt)] if system_prompt else []
        )
        if input_image_bytes is None or len(input_image_bytes) == 0:
            conversations = [
                Conversation(
                    messages=(
                        system_messages + [Message(role=Role.USER, content=content)]
                    )
                )
                for content in inputs
            ]
        else:
            conversations = [
                Conversation(
                    messages=(
                        system_messages
                        + [
                            Message(
                                role=Role.USER,
                                content=(
                                    [
                                        ContentItem(
                                            type=Type.IMAGE_BINARY, binary=image_bytes
                                        )
                                        for image_bytes in input_image_bytes
                                    ]
                                    + [ContentItem(type=Type.TEXT, content=content)]
                                ),
                            )
                        ]
                    )
                )
                for content in inputs
            ]

    generations = inference_engine.infer(
        input=conversations,
        inference_config=config,
    )
    return generations
