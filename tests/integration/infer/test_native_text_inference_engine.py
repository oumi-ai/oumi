import tempfile
from pathlib import Path

from lema.core.configs import ModelParams
from lema.core.types.turn import Conversation, Message, Role
from lema.inference import NativeTextInferenceEngine


def _get_default_model_params() -> ModelParams:
    return ModelParams(
        model_name="openai-community/gpt2",
        trust_remote_code=True,
    )


#
# Tests
#
def test_infer_online():
    engine = NativeTextInferenceEngine(_get_default_model_params())
    conversation = Conversation(
        messages=[
            Message(
                content="Hello world!",
                role=Role.USER,
            ),
            Message(
                content="Hello again!",
                role=Role.USER,
            ),
        ],
        metadata={"foo": "bar"},
        conversation_id="123",
    )
    expected_result = [
        Conversation(
            messages=[
                *conversation.messages,
                Message(
                    content="The first time I saw",
                    role=Role.ASSISTANT,
                ),
            ],
            metadata={"foo": "bar"},
            conversation_id="123",
        )
    ]
    result = engine.infer_online([conversation], max_new_tokens=5)
    assert expected_result == result


def test_infer_online_empty():
    engine = NativeTextInferenceEngine(_get_default_model_params())
    result = engine.infer_online([], max_new_tokens=5)
    assert [] == result


def test_infer_batch():
    with tempfile.TemporaryDirectory() as output_temp_dir:
        engine = NativeTextInferenceEngine(_get_default_model_params())
        conversation_1 = Conversation(
            messages=[
                Message(
                    content="Hello world!",
                    role=Role.USER,
                ),
                Message(
                    content="Hello again!",
                    role=Role.USER,
                ),
            ],
            metadata={"foo": "bar"},
            conversation_id="123",
        )
        conversation_2 = Conversation(
            messages=[
                Message(
                    content="Touche!",
                    role=Role.USER,
                ),
            ],
            metadata={"umi": "bar"},
            conversation_id="133",
        )
        expected_result = [
            Conversation(
                messages=[
                    *conversation_1.messages,
                    Message(
                        content="The first time I saw",
                        role=Role.ASSISTANT,
                    ),
                ],
                metadata={"foo": "bar"},
                conversation_id="123",
            ),
            Conversation(
                messages=[
                    *conversation_2.messages,
                    Message(
                        content="The U.S.",
                        role=Role.ASSISTANT,
                    ),
                ],
                metadata={"umi": "bar"},
                conversation_id="133",
            ),
        ]

        output_path = Path(output_temp_dir) / "b" / "output.jsonl"
        result = engine.infer_batch(
            [conversation_1, conversation_2],
            max_new_tokens=5,
            output_filepath=str(output_path),
        )
        assert result is None
        with open(output_path) as f:
            parsed_conversations = []
            for line in f:
                parsed_conversations.append(Conversation.model_validate_json(line))
            assert expected_result == parsed_conversations
