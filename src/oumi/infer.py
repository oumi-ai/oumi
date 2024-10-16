import argparse
from typing import List, Optional

from oumi.core.configs import InferenceConfig, InferenceEngineType
from oumi.core.inference import BaseInferenceEngine
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.inference import (
    AnthropicInferenceEngine,
    LlamaCppInferenceEngine,
    NativeTextInferenceEngine,
    RemoteInferenceEngine,
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
    elif config.engine == InferenceEngineType.ANTHROPIC:
        return AnthropicInferenceEngine(config.model)
    elif config.engine == InferenceEngineType.REMOTE:
        return RemoteInferenceEngine(config.model)
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
        "--interactive",
        action="store_true",
    )
    args, unknown = parser.parse_known_args()
    return args.config, args.interactive, unknown


def main():
    """Main entry point for running inference using Oumi.

    Training arguments are fetched from the following sources, ordered by
    decreasing priority:
    1. [Optional] Arguments provided as CLI arguments, in dotfile format
    2. [Optional] Arguments provided in a yaml config file
    3. Default arguments values defined in the data class
    """
    # Load configuration
    config_path, interactive, arg_list = parse_cli()

    config: InferenceConfig = InferenceConfig.from_yaml_and_arg_list(
        config_path, arg_list, logger=logger
    )
    config.validate()

    # Run inference
    infer_interactive(config)


def infer_interactive(config: InferenceConfig) -> None:
    """Interactively provide the model response for a user-provided input."""
    input_text = input("Enter your input prompt: ")
    model_response = infer(
        config=config,
        inputs=[
            input_text,
        ],
    )
    print(model_response[0])


# TODO: Consider stripping a prompt i.e., keep just newly generated tokens.
def infer(
    config: InferenceConfig,
    inputs: Optional[List[str]] = None,
) -> List[str]:
    """Runs batch inference for a model using the provided configuration.

    Args:
        config: The configuration to use for inference.
        inputs: A list of inputs for inference.

    Returns:
        object: A list of model responses.
    """
    inference_engine = _get_engine(config)
    # Pass None if no conversations are provided.
    conversations = None
    if inputs is not None and len(inputs) > 0:
        conversations = [
            Conversation(messages=[Message(content=content, role=Role.USER)])
            for content in inputs
        ]
    generations = inference_engine.infer(
        input=conversations,
        generation_params=config.generation,
    )
    if not generations:
        raise RuntimeError("No generations were returned.")
    return [conversation.messages[-1].content or "" for conversation in generations]


if __name__ == "__main__":
    main()
