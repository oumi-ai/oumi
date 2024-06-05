import argparse
from typing import cast

from omegaconf import OmegaConf

from lema.builders import (
    build_model,
    build_tokenizer,
)
from lema.core.types import InferenceConfig


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

    # Start with dataclass default values and type annotations
    all_configs = [OmegaConf.structured(InferenceConfig)]

    # Override with configuration file if provided
    if config_path is not None:
        all_configs.append(InferenceConfig.from_yaml(config_path))

    # Override with CLI arguments if provided
    all_configs.append(OmegaConf.from_cli(arg_list))

    # Merge and validate configs
    config = OmegaConf.to_object(OmegaConf.merge(*all_configs))
    if not isinstance(config, InferenceConfig):
        raise TypeError("config is not InferenceConfig")

    #
    # Run inference
    #
    infer_interactive(cast(InferenceConfig, config))


def infer_interactive(config: InferenceConfig) -> None:
    """Interactively provide the model response for a user-provided input."""
    input_text = input("Enter your input prompt: ")
    outputs_decoded = infer(
        config,
        [
            input_text,
        ],
    )
    print(outputs_decoded[0])


# TODO: Support writing predictions to files.
# TODO: Consider stripping a prompt i.e., keep just newly generated tokens.
def infer(config: InferenceConfig, input_batch):
    """Run batch inference for a model, using the provided configuration."""
    tokenizer = build_tokenizer(config.model)
    model = build_model(config)

    # Tokenization of input_batch.
    input_batch_tokenized = tokenizer(input_batch, return_tensors="pt")

    # Generate model outputs.
    model_device = next(model.parameters()).device
    input_batch_tokenized = input_batch_tokenized.to(model_device)
    outputs = model.generate(
        **input_batch_tokenized, max_new_tokens=config.generation.max_new_tokens
    )

    # Decode the outputs.
    outputs_decoded = []
    for input_idx in range(outputs.data.size(dim=0)):
        output = "".join(f"{tokenizer.decode(id)}" for id in outputs.data[input_idx])
        outputs_decoded.append(output)

    return outputs_decoded


if __name__ == "__main__":
    main()
