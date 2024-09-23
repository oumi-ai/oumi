import json
from pathlib import Path
from typing import List, Optional

import typer

from oumi.builders.data import build_dataset
from oumi.core.configs import JudgeConfig
from oumi.core.datasets import BaseLMSftDataset
from oumi.core.registry import REGISTRY
from oumi.core.types.turn import Conversation, Role
from oumi.judges.base_judge import Judge


def judge_dataset(
    config: JudgeConfig, dataset: BaseLMSftDataset
) -> List[Optional[str]]:
    """Judge a dataset."""
    judge = Judge(config)
    conversations = [dataset.conversation(idx) for idx in range(len(dataset))]
    judged_conversations = judge.judge(conversations)
    judge_messages = [
        conversation.last_message(Role.ASSISTANT)
        for conversation in judged_conversations
    ]
    return [message.content for message in judge_messages if message is not None]


def judge_conversations(
    config: JudgeConfig, conversations: List[Conversation]
) -> List[Optional[str]]:
    """Judge a list of conversations."""
    judge = Judge(config)
    judged_conversations = judge.judge(conversations)
    judge_messages = [
        conversation.last_message(Role.ASSISTANT)
        for conversation in judged_conversations
    ]
    return [message.content for message in judge_messages if message is not None]


def main(
    config_path: str = typer.Option(
        ..., "--config_path", help="Path to the judge config file"
    ),
    config_name: str = typer.Option(
        ...,
        "--config_name",
        help="Name of the judge configuration",
    ),
    input_file: Optional[str] = typer.Option(
        None, ..., "--input", help="Path to the input file (jsonl)"
    ),
    output_file: Optional[str] = typer.Option(
        None, "--output", help="Path to the output file (jsonl)"
    ),
    input_dataset: Optional[str] = typer.Option(
        None, "--dataset", help="Name of the dataset from the registry"
    ),
):
    """Judge a Oumi dataset or list of Oumi conversations."""
    # Load config
    if bool(config_name) == bool(config_path):
        raise ValueError(
            "Exactly one of 'config_name' or 'config_path' must be provided."
        )

    if bool(input_dataset) == bool(input_file):
        raise ValueError(
            "Exactly only one of 'input_dataset' or 'input_file' must be provided."
        )

    # Load judge config
    if config_name:
        judge_config_builder = REGISTRY.get_judge_config(config_name)
        if judge_config_builder is None:
            raise ValueError(f"Judge config '{config_name}' not found in registry.")
        judge_config = judge_config_builder()

    else:  # config_path is provided
        if not Path(config_path).exists():
            raise ValueError(f"Config file not found: '{config_path}'")
        judge_config = JudgeConfig.from_yaml(config_path)

    # Load judge inputs
    if input_file is not None:
        with open(input_file) as f:
            input_data = json.load(f)

        conversations = [Conversation(**conv) for conv in input_data]
        results = judge_conversations(judge_config, conversations=conversations)

    elif input_dataset is not None:
        dataset = build_dataset(dataset_name=input_dataset, tokenizer=None)
        if not isinstance(dataset, BaseLMSftDataset):
            raise ValueError(
                f"Dataset '{input_dataset}' is not an instance of BaseLMSftDataset. "
                "Please provide a valid dataset for judging."
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
