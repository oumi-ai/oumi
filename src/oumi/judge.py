from typing import Any, Dict, List, Optional, Union

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


def judge(
    config: JudgeConfig,
    data: Union[Dict[str, Any], List[Dict[str, Any]]],
    attributes: List[str],
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Judge a single data point or a list of data points."""
    if isinstance(data, dict):
        return judge_dataset(config, [data], attributes)[0]
    elif isinstance(data, list):
        return judge_dataset(config, data, attributes)
    else:
        raise ValueError("Data must be a dictionary or a list of dictionaries.")


def judge_conversation(
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


if __name__ == "__main__":
    test()
