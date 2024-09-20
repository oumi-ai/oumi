import argparse
from typing import List

from oumi.core.configs import GenerationConfig, InferenceConfig, ModelParams
from oumi.core.types.turn import Conversation, Message, Role
from oumi.inference import NativeTextInferenceEngine
from oumi.utils.logging import logger


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
        model_params=config.model,
        generation_config=config.generation,
        input=[
            input_text,
        ],
    )
    print(model_response[0][0])


# TODO: Support writing predictions to files.
# TODO: Consider stripping a prompt i.e., keep just newly generated tokens.
def infer(
    model_params: ModelParams,
    generation_config: GenerationConfig,
    input: List[str],
) -> List[str]:
    """Runs batch inference for a model using the provided configuration.

    Args:
        model_params: The configuration object containing the model parameters.
        generation_config: The configuration object for model generation.
        input: A list of text prompts of shape (num_batches, batch_size).
        exclude_prompt_from_response: Whether to trim the model's response and remove
          the prepended prompt.

    Returns:
        object: A list of model responses of shape (num_batches, batch_size).
    """
    inference_engine = NativeTextInferenceEngine(model_params)
    conversations = [
        Conversation(messages=[Message(content=content, role=Role.USER)])
        for content in input
    ]
    generations = inference_engine.infer(
        input=conversations,
        generation_config=generation_config,
    )
    if not generations:
        raise RuntimeError("No generations were returned.")
    return [conversation.messages[-1].content or "" for conversation in generations]


if __name__ == "__main__":
    main()
