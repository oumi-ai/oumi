from oumi.core.types.turn import Conversation, Message, Role
from oumi.judges.base_judge import Judge
from oumi.judges.judge_zoo import _get_default_local_judge_config


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
