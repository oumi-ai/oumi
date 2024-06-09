import argparse
from typing import List

from tqdm import tqdm

from lema.builders import (
    build_model,
    build_tokenizer,
)
from lema.core.types import GenerationConfig, InferenceConfig, ModelParams
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

    # Run inference
    infer_interactive(config)


def infer_interactive(config: InferenceConfig) -> None:
    """Interactively provide the model response for a user-provided input."""
    input_text = input("Enter your input prompt: ")
    model_response = infer(
        model_params=config.model,
        generation_config=config.generation,
        input=[
            [
                input_text,
            ],
        ],
    )
    print(model_response[0][0])


# TODO: Support writing predictions to files.
# TODO: Consider stripping a prompt i.e., keep just newly generated tokens.
def infer(
    model_params: ModelParams,
    generation_config: GenerationConfig,
    input: List[List[str]],
) -> List[List[str]]:
    """Run batch inference for a model, using the provided configuration.

    Args:
        model_params: The configuration object containing the model parameters.
        generation_config: The configuration object for model generation.
        input: A list of text prompts of shape (num_batches, batch_size).

    Returns:
        object: A list of model responses of shape (num_batches, batch_size).
    """
    tokenizer = build_tokenizer(model_params)
    model = build_model(model_params)
    model_device = next(model.parameters()).device

    # Tokenization of input (in place, batch mode).
    for batch_index, batch in enumerate(input):
        batch_tokenized = tokenizer(batch, return_tensors="pt", padding=True)
        batch_tokenized = batch_tokenized.to(model_device)
        input[batch_index] = batch_tokenized

    # Generate model outputs (batch mode).
    output = []
    for batch_index in tqdm(range(len(input)), desc="Generating Model Responses"):
        batch = input[batch_index]
        output.append(
            model.generate(**batch, max_new_tokens=generation_config.max_new_tokens)
        )

    # Decode the outputs (batch mode).
    output_decoded = []
    for batch in output:
        output_decoded.append(
            tokenizer.batch_decode(
                batch.data, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
        )

    return output_decoded


if __name__ == "__main__":
    main()
