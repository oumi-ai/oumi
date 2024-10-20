import json
from pathlib import Path
from typing import List, Optional

import jsonlines
import typer
from typing_extensions import Annotated

from oumi.builders.inference_engines import build_inference_engine
from oumi.core.cli import cli_utils
from oumi.core.configs import InferenceConfig, JudgeConfig
from oumi.core.registry import REGISTRY
from oumi.core.types.conversation import Conversation
from oumi.judge import judge_conversations, judge_dataset
from oumi.utils.io_utils import load_jsonlines


def _load_judge_config(config: str, extra_args: List[str]) -> JudgeConfig:
    judge_config_builder = REGISTRY.get_judge_config(config)

    if judge_config_builder:
        if extra_args:
            typer.echo(
                "For consistent judge results, a named judge config cannot be "
                f"overridden with extra arguments. Got: {extra_args}. "
                "Please register a new named judge config, or provide a path to a "
                "judge config file."
            )
            raise typer.Exit(code=1)
        return judge_config_builder()

    if not Path(config).exists():
        typer.echo(f"Config file not found: '{config}'")
        raise typer.Exit(code=1)

    return JudgeConfig.from_yaml_and_arg_list(config, extra_args)


def dataset(
    ctx: typer.Context,
    config: Annotated[
        str, typer.Option(*cli_utils.CONFIG_FLAGS, help="Path to the judge config file")
    ],
    dataset_name: Annotated[
        Optional[str], typer.Option(help="Name of the dataset from the registry")
    ] = None,
    dataset_subset: Annotated[
        Optional[str], typer.Option(help="Subset of the dataset to use, if applicable")
    ] = None,
    dataset_split: Annotated[
        Optional[str], typer.Option(help="Split of the dataset to use.")
    ] = "train",
    output_file: Annotated[
        Optional[str], typer.Option(help="Path to the output file (jsonl)")
    ] = None,
):
    """Judge a dataset."""
    if not dataset_name:
        typer.echo("Dataset name is required.")
        raise typer.Exit(code=1)

    # Load the judge config
    extra_args = cli_utils.parse_extra_cli_args(ctx)

    judge_config = _load_judge_config(config, extra_args)

    # Load the dataset class from the registry
    dataset_class = REGISTRY.get_dataset(dataset_name, subset=dataset_subset)

    if dataset_class is None:
        typer.echo(f"Dataset '{dataset_name}' not found in registry.")
        raise typer.Exit(code=1)

    dataset = dataset_class(
        split=dataset_split,
        subset=dataset_subset,
    )

    # Judge the dataset
    results = judge_dataset(judge_config, dataset=dataset)

    # Save the results
    if output_file:
        typer.echo(f"Saving results to {output_file}")
        with jsonlines.open(output_file, mode="w") as writer:
            writer.write_all(results)
    else:
        for result in results:
            print(json.dumps(result))


def conversations(
    ctx: typer.Context,
    config: Annotated[
        str, typer.Option(*cli_utils.CONFIG_FLAGS, help="Path to the judge config file")
    ],
    input_file: Annotated[
        Optional[str], typer.Option(help="Path to the input file (jsonl)")
    ] = None,
    output_file: Annotated[
        Optional[str], typer.Option(help="Path to the output file (jsonl)")
    ] = None,
):
    """Judge a list of conversations."""
    # Load the judge config
    extra_args = cli_utils.parse_extra_cli_args(ctx)

    judge_config = _load_judge_config(config, extra_args)

    # Load the conversations from the input file
    if not input_file:
        typer.echo("Input file is required.")
        raise typer.Exit(code=1)

    input_data = load_jsonlines(input_file)
    conversations = [Conversation.from_dict(conv) for conv in input_data]

    # Judge the conversations
    results = judge_conversations(judge_config, judge_inputs=conversations)

    # Save the results
    if output_file:
        typer.echo(f"Saving results to {output_file}")
        with jsonlines.open(output_file, mode="w") as writer:
            writer.write_all(results)
    else:
        for result in results:
            print(json.dumps(result))


def model(
    ctx: typer.Context,
    config: Annotated[
        str, typer.Option(*cli_utils.CONFIG_FLAGS, help="Path to the judge config file")
    ],
    inference_config: Annotated[
        str,
        typer.Option(*cli_utils.CONFIG_FLAGS, help="Path to the inference config file"),
    ],
    input_file: Annotated[
        Optional[str], typer.Option(help="Path to the input file (jsonl)")
    ] = None,
    output_file: Annotated[
        Optional[str], typer.Option(help="Path to the output file (jsonl)")
    ] = None,
):
    """Judge the outputs of a model on a dataset."""
    # Load the judge config
    judge_extra_args = cli_utils.parse_extra_cli_args(ctx)
    judge_config = _load_judge_config(config, judge_extra_args)

    # Load the inference config
    inference_extra_args = cli_utils.parse_extra_cli_args(ctx)
    model_inference_config: InferenceConfig = InferenceConfig.from_yaml_and_arg_list(
        inference_config, inference_extra_args
    )

    if not model_inference_config.engine:
        typer.echo("Inference engine is required.")
        raise typer.Exit(code=1)

    # Load the dataset
    if not input_file:
        typer.echo("Input file is required.")
        raise typer.Exit(code=1)

    input_data = load_jsonlines(input_file)
    input_conversations = [Conversation.from_dict(output) for output in input_data]

    # Run inference
    inference_engine = build_inference_engine(
        model_inference_config.engine, model_params=model_inference_config.model
    )

    model_outputs = inference_engine.infer(
        input=input_conversations, generation_params=model_inference_config.generation
    )

    results = judge_conversations(judge_config, judge_inputs=model_outputs)

    if output_file:
        typer.echo(f"Saving results to {output_file}")
        with jsonlines.open(output_file, mode="w") as writer:
            writer.write_all(results)
    else:
        for result in results:
            print(json.dumps(result))
