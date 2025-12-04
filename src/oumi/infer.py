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


def get_engine(config: InferenceConfig) -> BaseInferenceEngine:
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


def infer(
    config: InferenceConfig,
    inputs: Optional[list[str]] = None,
    inference_engine: Optional[BaseInferenceEngine] = None,
    *,
    input_image_bytes: Optional[list[bytes]] = None,
    system_prompt: Optional[str] = None,
) -> list[Conversation]:
    """Infer using the given configuration and inputs.

    Args:
        config: The inference configuration.
        inputs: The inputs to use for inference. If not provided, then the
            input provided within the config will be used instead.
        inference_engine: The inference engine to use for inference. If not
            provided, then a new inference engine will be created based on the
            config.
        input_image_bytes: The image bytes to use for multimodal model
            inference.
        system_prompt: System prompt for task-specific instructions.

    Returns:
        A list of conversations with assistant responses.
    """
    if inference_engine is None:
        inference_engine = get_engine(config)

    if inputs is None:
        # Use the input_filepath from the configuration
        # For now, just return empty list
        return []

    conversations = []
    for input_text in inputs:
        # Create conversation with system message (if provided) and user message
        messages = []
        if system_prompt:
            messages.append(Message(role=Role.SYSTEM, content=system_prompt))

        # Handle multimodal input
        if input_image_bytes:
            content_items = []
            for image_bytes in input_image_bytes:
                content_items.append(
                    ContentItem(type=Type.IMAGE_URL, content=image_bytes)
                )
            content_items.append(ContentItem(type=Type.TEXT, content=input_text))
            messages.append(Message(role=Role.USER, content=content_items))
        else:
            messages.append(Message(role=Role.USER, content=input_text))

        conversation = Conversation(messages=messages)
        conversations.append(conversation)

    model_response = inference_engine.infer(
        input=conversations,
        inference_config=config,
    )

    return model_response
