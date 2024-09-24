import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer

from oumi.core.configs import JudgeConfig
from oumi.core.datasets import BaseLMSftDataset
from oumi.core.registry import REGISTRY
from oumi.core.types.turn import Conversation
from oumi.judges.oumi_judge import OumiXmlJudge as Judge


def judge_dataset(
    config: JudgeConfig, dataset: BaseLMSftDataset
) -> List[Dict[str, Any]]:
    """Judge a dataset."""
    judge = Judge(config)
    judge_inputs = [dataset.conversation(idx) for idx in range(len(dataset))]
    judge_outputs = judge.judge(judge_inputs)
    return judge_outputs


def judge_conversations(
    config: JudgeConfig, judge_inputs: List[Conversation]
) -> List[Dict[str, Any]]:
    """Judge a list of conversations."""
    judge = Judge(config)
    judge_outputs = judge.judge(judge_inputs)
    return judge_outputs


def main(
    config_path: Optional[str] = typer.Option(
        default=None, help="Path to the judge config file"
    ),
    config_name: Optional[str] = typer.Option(
        default=None,
        help="Name of the judge configuration",
    ),
    input_file: Optional[str] = typer.Option(
        default=None, help="Path to the input file (jsonl)"
    ),
    output_file: Optional[str] = typer.Option(
        default=None, help="Path to the output file (jsonl)"
    ),
    dataset_name: Optional[str] = typer.Option(
        default=None, help="Name of the dataset from the registry"
    ),
    dataset_subset: Optional[str] = typer.Option(
        default=None, help="Subset of the dataset to use, if applicable"
    ),
    dataset_split: Optional[str] = typer.Option(
        default="train",
        help="Split of the dataset to use.",
    ),
):
    """Judge a Oumi dataset or list of Oumi conversations."""
    # Load config
    if bool(config_name) == bool(config_path):
        raise ValueError(
            "Exactly one of 'config_name' or 'config_path' must be provided."
        )

    if bool(dataset_name) == bool(input_file):
        raise ValueError(
            "Exactly only one of 'input_dataset' or 'input_file' must be provided."
        )

    # Load judge config
    if config_name:
        judge_config_builder = REGISTRY.get_judge_config(config_name)
        if judge_config_builder is None:
            raise ValueError(f"Judge config '{config_name}' not found in registry.")
        judge_config = judge_config_builder()

    elif config_path:
        if not Path(config_path).exists():
            raise ValueError(f"Config file not found: '{config_path}'")
        judge_config = JudgeConfig.from_yaml(config_path)

    # Load judge inputs
    if input_file is not None:
        with open(input_file) as f:
            input_data = json.load(f)

        conversations = [Conversation(**conv) for conv in input_data]
        results = judge_conversations(judge_config, judge_inputs=conversations)

    elif dataset_name is not None:
        dataset_class = REGISTRY.get_dataset(dataset_name, subset=dataset_subset)

        if dataset_class is None:
            raise ValueError(f"Dataset '{dataset_name}' not found in registry.")
        dataset = dataset_class(
            split=dataset_split,
            subset=dataset_subset,
        )

        results = judge_dataset(judge_config, dataset=dataset)

    # Output
    if output_file:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    typer.run(main)
