import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import jsonlines
import typer

from oumi.core.configs import JudgeConfig
from oumi.core.datasets import BaseLMSftDataset
from oumi.core.registry import REGISTRY
from oumi.core.types.turn import Conversation
from oumi.judges.oumi_judge import OumiXmlJudge as Judge


def judge_dataset(
    config: JudgeConfig, dataset: BaseLMSftDataset
) -> List[Dict[str, Any]]:
    """Judge a dataset.

    This function evaluates a given dataset using a specified Judge configuration.

    The function performs the following steps:
    1. Initializes the Judge with the provided configuration.
    2. Iterates through the dataset to extract conversation inputs.
    3. Uses the Judge to evaluate each conversation input.
    4. Collects and returns the judged outputs.

    Args:
        config: The configuration for the judge.
        dataset: The dataset to be judged. This dataset
            should be compatible with the Supervised Finetuning Dataset class.

    Returns:
        List[Dict[str, Any]]: A list of judgement results for each conversation.

        Example output:
            [
                {'helpful': True, 'safe': False},
                {'helpful': True, 'safe': True},
            ]

    Example:
        >>> config = JudgeConfig(...)
        >>> dataset = SomeDataset(...)
        >>> judged_outputs = judge_dataset(config, dataset)
        >>> for output in judged_outputs:
        ...     print(output)
    """
    judge = Judge(config)
    judge_inputs = [dataset.conversation(idx) for idx in range(len(dataset))]
    judge_outputs = judge.judge(judge_inputs)
    return judge_outputs


def judge_conversations(
    config: JudgeConfig, judge_inputs: List[Conversation]
) -> List[Dict[str, Any]]:
    """Judge a list of conversations.

    This function evaluates a list of conversations using the specified Judge.

    The function performs the following steps:
    1. Initializes the Judge with the provided configuration.
    2. Uses the Judge to evaluate each conversation input.
    3. Collects and returns the judged outputs.

    Args:
        config: The configuration for the judge.
        judge_inputs: A list of Conversation objects to be judged.

    Returns:
        List[Dict[str, Any]]: A list of judgement results for each conversation.

        Example output:
            [
                {'helpful': True, 'safe': False},
                {'helpful': True, 'safe': True},
            ]

    Example:
        >>> config = JudgeConfig(...)
        >>> judge_inputs = [Conversation(...), Conversation(...)]
        >>> judged_outputs = judge_conversations(config, judge_inputs)
        >>> for output in judged_outputs:
        ...     print(output)
    """
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
    """Main entry point for the judge script.

    Args:
        config_path (Optional[str]): Path to the judge config file.
        config_name (Optional[str]): Name of the judge configuration.
        input_file (Optional[str]): Path to the input file (jsonl).
        output_file (Optional[str]): Path to the output file (jsonl).
        dataset_name (Optional[str]): Name of the dataset from the registry.
        dataset_subset (Optional[str]): Subset of the dataset to use, if applicable.
        dataset_split (Optional[str]): Split of the dataset to use.

    Raises:
        ValueError: If both or neither of 'config_name' and 'config_path' are provided.
        ValueError: If both or neither of 'dataset_name' and 'input_file' are provided.
        ValueError: If the specified judge config or dataset is not found in the
            registry.
        ValueError: If the specified config file does not exist.

    Example:
        >>> main(
        ...     config_name="default_judge",
        ...     input_file="input.jsonl",
        ...     output_file="output.jsonl"
        ... )
    """
    # Load config
    if bool(config_name) == bool(config_path):
        raise ValueError(
            "Exactly one of 'config_name' or 'config_path' must be provided. "
            f"Currently: {'both' if config_name and config_path else 'neither'} "
            "specified."
        )

    if bool(dataset_name) == bool(input_file):
        raise ValueError(
            "Exactly one of 'dataset_name' or 'input_file' must be provided. "
            f"Currently: {'both' if dataset_name and input_file else 'neither'} "
            "specified."
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
        with jsonlines.open(output_file, mode="w") as writer:
            writer.write_all(results)
    else:
        for result in results:
            print(json.dumps(result))


if __name__ == "__main__":
    typer.run(main)
