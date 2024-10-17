import json
from pathlib import Path
from typing import Optional

import jsonlines
import typer
from typing_extensions import Annotated

from oumi.core.configs import JudgeConfig
from oumi.core.registry import REGISTRY
from oumi.core.types.conversation import Conversation
from oumi.judge import judge_conversations, judge_dataset


def _load_judge_config(
    config_name: Optional[str], config_path: Optional[str]
) -> JudgeConfig:
    if bool(config_name) == bool(config_path):
        raise ValueError(
            "Exactly one of 'config_name' or 'config_path' must be provided. "
            f"Currently: {'both' if config_name and config_path else 'neither'} "
            "specified."
        )

    if config_name:
        judge_config_builder = REGISTRY.get_judge_config(config_name)
        if judge_config_builder is None:
            raise ValueError(f"Judge config '{config_name}' not found in registry.")
        return judge_config_builder()

    if not config_path or not Path(config_path).exists():
        raise ValueError(f"Config file not found: '{config_path}'")
    return JudgeConfig.from_yaml(config_path)


def dataset(
    ctx: typer.Context,
    config_path: Annotated[
        Optional[str], typer.Option(help="Path to the judge config file")
    ] = None,
    config_name: Annotated[
        Optional[str], typer.Option(help="Name of the judge configuration")
    ] = None,
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
        raise ValueError("Dataset name is required.")

    judge_config = _load_judge_config(config_name, config_path)

    dataset_class = REGISTRY.get_dataset(dataset_name, subset=dataset_subset)

    if dataset_class is None:
        raise ValueError(f"Dataset '{dataset_name}' not found in registry.")

    dataset = dataset_class(
        split=dataset_split,
        subset=dataset_subset,
    )

    results = judge_dataset(judge_config, dataset=dataset)

    if output_file:
        with jsonlines.open(output_file, mode="w") as writer:
            writer.write_all(results)
    else:
        for result in results:
            print(json.dumps(result))


def conversations(
    ctx: typer.Context,
    config_path: Annotated[
        Optional[str], typer.Option(help="Path to the judge config file")
    ] = None,
    config_name: Annotated[
        Optional[str], typer.Option(help="Name of the judge configuration")
    ] = None,
    input_file: Annotated[
        Optional[str], typer.Option(help="Path to the input file (jsonl)")
    ] = None,
    output_file: Annotated[
        Optional[str], typer.Option(help="Path to the output file (jsonl)")
    ] = None,
):
    """Judge a list of conversations."""
    judge_config = _load_judge_config(config_name, config_path)

    if not input_file:
        raise ValueError("Input file is required.")

    with open(input_file) as f:
        input_data = json.load(f)

    conversations = [Conversation(**conv) for conv in input_data]

    results = judge_conversations(judge_config, judge_inputs=conversations)

    if output_file:
        with jsonlines.open(output_file, mode="w") as writer:
            writer.write_all(results)
    else:
        for result in results:
            print(json.dumps(result))
