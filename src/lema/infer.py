import argparse
from typing import List

from lema.core.configs import GenerationConfig, InferenceConfig, ModelParams
from lema.inference import NativeTextInferenceEngine
from lema.utils.logging import logger


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
    """Main entry point for running inference using LeMa.

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
    exclude_prompt_from_response: bool = True,
    batch_size: int = 2,
) -> List[str]:
    """Runs batch inference for a model using the provided configuration.

    Args:
        model_params: The configuration object containing the model parameters.
        generation_config: The configuration object for model generation.
        input: A list of text prompts of shape (num_batches, batch_size).
        exclude_prompt_from_response: Whether to trim the model's response and remove
          the prepended prompt.
        batch_size: The number of sequences to generate in parallel.

    Returns:
        object: A list of model responses of shape (num_batches, batch_size).
    """
    inference_engine = NativeTextInferenceEngine(model_params)
    generations = inference_engine.infer(
        input,
        max_new_tokens=generation_config.max_new_tokens,
        exclude_prompt_from_response=exclude_prompt_from_response,
        batch_size=batch_size,
    )
    return [generation.messages[-1].content for generation in generations]


if __name__ == "__main__":
    main()
