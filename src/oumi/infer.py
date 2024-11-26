import argparse
from typing import Optional

from oumi.core.configs import InferenceConfig, InferenceEngineType
from oumi.core.inference import BaseInferenceEngine
from oumi.core.types.conversation import Conversation, Message, Role, Type
from oumi.inference import (
    AnthropicInferenceEngine,
    LlamaCppInferenceEngine,
    NativeTextInferenceEngine,
    RemoteInferenceEngine,
    RemoteVLLMInferenceEngine,
    SGLangInferenceEngine,
    VLLMInferenceEngine,
)
from oumi.utils.image_utils import load_image_png_bytes_from_path
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


def parse_cli():
    """Parses command line arguments and returns the configuration filename."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", default=None, help="Path to the configuration file"
    )
    parser.add_argument(
        "-i",
        "--image",
        type=argparse.FileType("rb"),
        help="File path of an input image to be used with `image+text` VLLMs.",
    )
    args, unknown = parser.parse_known_args()
    return args.config, args.image, unknown


def main():
    """Main entry point for running inference using Oumi.

    Training arguments are fetched from the following sources, ordered by
    decreasing priority:
    1. [Optional] Arguments provided as CLI arguments, in dotfile format
    2. [Optional] Arguments provided in a yaml config file
    3. Default arguments values defined in the data class
    """
    # Load configuration
    config_path, input_image_filepath, arg_list = parse_cli()

    config: InferenceConfig = InferenceConfig.from_yaml_and_arg_list(
        config_path, arg_list, logger=logger
    )
    config.validate()

    input_image_png_bytes: Optional[bytes] = (
        load_image_png_bytes_from_path(input_image_filepath)
        if input_image_filepath
        else None
    )

    # Run inference
    infer_interactive(config, input_image_bytes=input_image_png_bytes)


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

    image_messages = (
        [
            Message(
                binary=input_image_bytes,
                type=Type.IMAGE_BINARY,
                role=Role.USER,
            )
        ]
        if input_image_bytes is not None
        else []
    )

    # Pass None if no conversations are provided.
    conversations = None
    if inputs is not None and len(inputs) > 0:
        conversations = [
            Conversation(
                messages=(image_messages + [Message(content=content, role=Role.USER)])
            )
            for content in inputs
        ]
    generations = inference_engine.infer(
        input=conversations,
        inference_config=config,
    )
    return generations


if __name__ == "__main__":
    main()
