import json
from typing import Any, Dict, List, Optional

import typer

from oumi.builders.data import build_dataset
from oumi.core.configs import JudgeConfig
from oumi.core.types.turn import Conversation, Message, Role
from oumi.judges.base_judge import Judge
from oumi.judges.judge_zoo import _get_default_local_judge_config


def judge_dataset(
    config: JudgeConfig, dataset: List[Dict[str, Any]], attributes: List[str]
) -> List[Dict[str, Any]]:
    """Judge a dataset using a model."""
    judge = Judge(config)
    conversations = [Conversation(**data) for data in dataset]
    judged_conversations = judge.judge(conversations)
    return [conv.model_dump() for conv in judged_conversations]


def judge_conversations(
    config: JudgeConfig, conversations: List[Conversation]
) -> List[Optional[str]]:
    """Judge a single conversation."""
    judge = Judge(config)
    judged_conversations = judge.judge(conversations)
    judge_messages = [
        conversation.last_message(Role.ASSISTANT)
        for conversation in judged_conversations
    ]
    return [message.content for message in judge_messages if message is not None]


def test():
    """Tests the Judge class."""
    # Create a Judge instance
    judge = Judge(_get_default_local_judge_config())

    # Create a sample conversation
    conversation = Conversation(
        messages=[
            Message(role=Role.SYSTEM, content="You are a helpful assistant."),
            Message(role=Role.USER, content="What is the capital of France?"),
            Message(role=Role.ASSISTANT, content="The capital of France is Paris."),
        ]
    )

    # Judge the conversation
    judgements = judge.judge([conversation])

    print("\nJudgements:")
    for judgement in judgements:
        print(f"Attribute: {judgement.metadata['judge_attribute_name']}")
        print(f"Raw Judgement: {judgement.messages[-1].content}")
        print(f"Parsed Judgement: {judgement.metadata['parsed_judgement']}")
        print()


def main(
    config_path: str = typer.Option(
        ..., "--config", help="Path to the judge config file"
    ),
    input_file: Optional[str] = typer.Option(
        ..., "--input", help="Path to the input file (jsonl)"
    ),
    output_file: Optional[str] = typer.Option(
        None, "--output", help="Path to the output file (jsonl)"
    ),
    dataset_name: Optional[str] = typer.Option(
        ..., "--dataset", help="Name of the dataset from the registry"
    ),
):
    """Judge a Oumi dataset or list of Oumi conversations."""
    # Load config
    judge_config = JudgeConfig.from_yaml(config_path)

    if input_file is not None:
        with open(input_file) as f:
            input_data = json.load(f)

        conversations = [Conversation(**conv) for conv in input_data]
        results = judge_conversations(judge_config, conversations=conversations)

    elif dataset_name is not None:
        dataset = build_dataset(dataset_name=dataset_name, tokenizer=None)
        results = judge_dataset(judge_config, dataset=dataset)

    else:
        typer.echo(
            "Error: Either --input or --dataset must be specified.",
            err=True,
        )
        raise typer.Exit(code=1)

    if output_file:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    typer.run(main)
