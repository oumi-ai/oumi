from typing import Optional

from oumi.core.configs import InferenceConfig, InferenceEngineType
from oumi.core.inference import BaseInferenceEngine
from oumi.core.types.conversation import (
    Conversation,
    Message,
    MessageContentItem,
    Role,
    Type,
)
from oumi.inference import (
    AnthropicInferenceEngine,
    LlamaCppInferenceEngine,
    NativeTextInferenceEngine,
    RemoteInferenceEngine,
    RemoteVLLMInferenceEngine,
    SGLangInferenceEngine,
    VLLMInferenceEngine,
)
from oumi.utils.logging import logger


def _get_engine(config: InferenceConfig) -> BaseInferenceEngine:
    """Returns the inference engine based on the provided config."""
    if config.engine is None:
        logger.warning(
            "No inference engine specified. Using the default 'native' engine."
        )
        return NativeTextInferenceEngine(config.model)
    elif config.engine == InferenceEngineType.NATIVE:
        return NativeTextInferenceEngine(config.model)
    elif config.engine == InferenceEngineType.VLLM:
        return VLLMInferenceEngine(config.model)
    elif config.engine == InferenceEngineType.LLAMACPP:
        return LlamaCppInferenceEngine(config.model)
    elif config.engine in (
        InferenceEngineType.REMOTE_VLLM,
        InferenceEngineType.SGLANG,
        InferenceEngineType.ANTHROPIC,
        InferenceEngineType.REMOTE,
    ):
        if config.remote_params is None:
            raise ValueError(
                "remote_params must be configured "
                f"for the '{config.engine}' inference engine in inference config."
            )
        if config.engine == InferenceEngineType.REMOTE_VLLM:
            return RemoteVLLMInferenceEngine(config.model, config.remote_params)
        elif config.engine == InferenceEngineType.SGLANG:
            return SGLangInferenceEngine(config.model, config.remote_params)
        elif config.engine == InferenceEngineType.ANTHROPIC:
            return AnthropicInferenceEngine(config.model, config.remote_params)
        else:
            assert config.engine == InferenceEngineType.REMOTE
            return RemoteInferenceEngine(config.model, config.remote_params)
    else:
        logger.warning(
            f"Unsupported inference engine: {config.engine}. "
            "Falling back to the default 'native' engine."
        )
        return NativeTextInferenceEngine(config.model)


def infer_interactive(
    config: InferenceConfig, *, input_image_bytes: Optional[bytes] = None
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
    input_image_bytes: Optional[bytes] = None,
) -> list[Conversation]:
    """Runs batch inference for a model using the provided configuration.

    Args:
        config: The configuration to use for inference.
        inputs: A list of inputs for inference.
        inference_engine: The engine to use for inference. If unspecified, the engine
            will be inferred from `config`.
        input_image_bytes: An input PNG image bytes to be used with `image+text` VLLMs.
            Only used in interactive mode.

    Returns:
        object: A list of model responses.
    """
    if not inference_engine:
        inference_engine = _get_engine(config)

    # Pass None if no conversations are provided.
    conversations = None
    if inputs is not None and len(inputs) > 0:
        if input_image_bytes is None:
            conversations = [
                Conversation(messages=[Message(role=Role.USER, content=content)])
                for content in inputs
            ]
        else:
            conversations = [
                Conversation(
                    messages=[
                        Message(
                            role=Role.USER,
                            content=[
                                MessageContentItem(
                                    type=Type.IMAGE_BINARY, binary=input_image_bytes
                                ),
                                MessageContentItem(type=Type.TEXT, content=content),
                            ],
                        ),
                    ]
                )
                for content in inputs
            ]

    generations = inference_engine.infer(
        input=conversations,
        inference_config=config,
    )
    return generations
