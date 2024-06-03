import argparse

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
    base_config = OmegaConf.structured(InferenceConfig)

    # Override with configuration file if provided
    if config_path is not None:
        file_config = OmegaConf.load(config_path)
        config = OmegaConf.merge(base_config, file_config)
    else:
        config = base_config

    # Override with CLI arguments if provided
    cli_config = OmegaConf.from_cli(arg_list)
    config = OmegaConf.merge(config, cli_config)

    # Merge and validate configs
    config: InferenceConfig = OmegaConf.to_object(config)

    #
    # Run inference
    #
    infer(config, interactive)


def infer(config: InferenceConfig, interactive: bool = False) -> None:
    """Evaluate a model using the provided configuration."""
    tokenizer = build_tokenizer(config.model)

    model = build_model(config)

    input_texts = []
    if interactive:
        input_text = input("Enter your input prompt: ")
        input_texts.append(input_text)
    else:
        # TODO: Support reading inputs from datasets.
        raise NotImplementedError("Non-interactive inference is not implemented yet")

    inputs = tokenizer(input_texts, return_tensors="pt")

    model_device = next(model.parameters()).device
    inputs = inputs.to(model_device)

    outputs = model.generate(**inputs, max_new_tokens=config.generation.max_new_tokens)

    # TODO: Support writing predictions to files.
    # TODO: Consider stripping a prompt i.e., keep just newly generated tokens.
    for input_idx in range(outputs.data.size(dim=0)):
        print(f"Prompt: {input_texts[input_idx]}")
        for token_id in outputs.data[input_idx]:
            print(f"| | {token_id:5d} | {tokenizer.decode(token_id):8s}")


if __name__ == "__main__":
    main()
