import argparse
from typing import List

from lema.builders import (
    build_model,
    build_tokenizer,
)
from lema.core.types import InferenceConfig
from lema.logging import logger


def parse_cli():
    """Parse command line arguments and return the configuration filename."""
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

    #
    # Run inference
    #
    infer_interactive(config)


def infer_interactive(config: InferenceConfig) -> None:
    """Interactively provide the model response for a user-provided input."""
    input_text = input("Enter your input prompt: ")
    model_response = infer(
        config,
        [
            [
                input_text,
            ],
        ],
    )
    print(model_response[0][0])


# TODO: Support writing predictions to files.
# TODO: Consider stripping a prompt i.e., keep just newly generated tokens.
def infer(config: InferenceConfig, input: List[List[str]]) -> List[List[str]]:
    """Run batch inference for a model, using the provided configuration.

    Args:
        config: The desired configuration for inference.
        input: A list of text prompts of shape (num_batches, batch_size).

    Returns:
        object: A list of model responses of shape (num_batches, batch_size).
    """
    tokenizer = build_tokenizer(config.model)
    model = build_model(config)
    model_device = next(model.parameters()).device

    # Tokenization of input (in place).
    for batch_index, batch in enumerate(input):
        batch_tokenized = tokenizer(batch, return_tensors="pt")
        batch_tokenized = batch_tokenized.to(model_device)
        input[batch_index] = batch_tokenized

    # Generate model outputs.
    output = []
    for batch in input:
        output.append(
            model.generate(**batch, max_new_tokens=config.generation.max_new_tokens)
        )

    # Decode the outputs.
    output_decoded = []
    for batch in output:
        batch_output_decoded = []
        for prompt_index in range(batch.data.size(dim=0)):
            response = "".join(tokenizer.decode(id) for id in batch.data[prompt_index])
            batch_output_decoded.append(response)
        output_decoded.append(batch_output_decoded)

    return output_decoded


if __name__ == "__main__":
    main()
