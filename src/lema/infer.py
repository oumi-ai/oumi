import argparse

from omegaconf import OmegaConf

from lema.builders import (    
    build_model,
    build_tokenizer,
)

from lema.core.types import InferenceConfig
from lema.core.types import TrainingConfig


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
    infer(config)


def infer(config: InferenceConfig) -> None:
    """Evaluate a model using the provided configuration."""
    train_config = TrainingConfig(model=config.model)    

    tokenizer = build_tokenizer(train_config)

    model = build_model(train_config)

    inputs = tokenizer(["Today is"], return_tensors="pt")

    model_device = next(model.parameters()).device
    inputs = inputs.to(model_device)

    outputs = model.generate(**inputs, max_new_tokens=10)    
        
    for tok in outputs.data[0]:    
        print(f"| {tok:5d} | {tokenizer.decode(tok):8s}")


if __name__ == "__main__":
    main()
