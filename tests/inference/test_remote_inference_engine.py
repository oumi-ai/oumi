import tempfile
import time
from pathlib import Path
from typing import List

import jsonlines
import pytest
from aioresponses import aioresponses

from oumi.core.configs import GenerationParams, ModelParams, RemoteParams
from oumi.core.types.conversation import Conversation, Message, Role, Type
from oumi.inference import RemoteInferenceEngine

_TARGET_SERVER: str = "http://fakeurl"


#
# Fixtures
#
@pytest.fixture
def mock_aioresponse():
    with aioresponses() as m:
        yield m


def _get_default_model_params() -> ModelParams:
    return ModelParams(
        model_name="openai-community/gpt2",
        trust_remote_code=True,
    )


def _setup_input_conversations(filepath: str, conversations: List[Conversation]):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    Path(filepath).touch()
    with jsonlines.open(filepath, mode="w") as writer:
        for conversation in conversations:
            json_obj = conversation.to_dict()
            writer.write(json_obj)
    # Add some empty lines into the file
    with open(filepath, "a") as f:
        f.write("\n\n\n")


#
# Tests
#
def test_infer_online():
    with aioresponses() as m:
        m.post(
            _TARGET_SERVER,
            status=200,
            payload=dict(
                choices=[
                    {
                        "message": {
                            "role": "assistant",
                            "content": "The first time I saw",
                        }
                    }
                ]
            ),
        )

        engine = RemoteInferenceEngine(_get_default_model_params())
        conversation = Conversation(
            messages=[
                Message(
                    content="Hello world!",
                    role=Role.USER,
                ),
                Message(
                    binary=b"Hello again!",
                    role=Role.USER,
                    type=Type.IMAGE_PATH,
                ),
                Message(
                    content="a url for our image",
                    role=Role.USER,
                    type=Type.IMAGE_URL,
                ),
                Message(
                    binary=b"a binary image",
                    role=Role.USER,
                    type=Type.IMAGE_BINARY,
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
        result = engine.infer_online(
            [conversation],
            GenerationParams(
                max_new_tokens=5, remote_params=RemoteParams(api_url=_TARGET_SERVER)
            ),
        )
        assert expected_result == result


def test_infer_no_remote_params():
    engine = RemoteInferenceEngine(_get_default_model_params())
    with pytest.raises(
        ValueError, match="Remote params must be provided in generation_params."
    ):
        engine.infer_online([], GenerationParams())
    with pytest.raises(
        ValueError, match="Remote params must be provided in generation_params."
    ):
        engine.infer_from_file("path", GenerationParams())


def test_infer_online_empty():
    engine = RemoteInferenceEngine(_get_default_model_params())
    expected_result = []
    result = engine.infer_online(
        [],
        GenerationParams(
            max_new_tokens=5, remote_params=RemoteParams(api_url=_TARGET_SERVER)
        ),
    )
    assert expected_result == result


def test_infer_online_fails():
    with aioresponses() as m:
        m.post(_TARGET_SERVER, status=401)
        m.post(_TARGET_SERVER, status=401)
        m.post(_TARGET_SERVER, status=401)
        m.post(_TARGET_SERVER, status=501)

        engine = RemoteInferenceEngine(_get_default_model_params())
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
        with pytest.raises(RuntimeError, match="Failed to query API after 3 retries."):
            _ = engine.infer_online(
                [conversation],
                GenerationParams(
                    max_new_tokens=5, remote_params=RemoteParams(api_url=_TARGET_SERVER)
                ),
            )


def test_infer_online_recovers_from_retries():
    with aioresponses() as m:
        m.post(_TARGET_SERVER, status=500)
        m.post(
            _TARGET_SERVER,
            status=200,
            payload=dict(
                choices=[
                    {
                        "message": {
                            "role": "assistant",
                            "content": "The first time I saw",
                        }
                    }
                ]
            ),
        )

        engine = RemoteInferenceEngine(_get_default_model_params())
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
        result = engine.infer_online(
            [conversation],
            GenerationParams(
                max_new_tokens=5, remote_params=RemoteParams(api_url=_TARGET_SERVER)
            ),
        )
        assert expected_result == result


def test_infer_online_multiple_requests():
    with aioresponses() as m:
        m.post(
            _TARGET_SERVER,
            status=200,
            payload=dict(
                choices=[
                    {
                        "message": {
                            "role": "assistant",
                            "content": "The first time I saw",
                        }
                    }
                ]
            ),
        )
        m.post(
            _TARGET_SERVER,
            status=200,
            payload=dict(
                choices=[
                    {
                        "message": {
                            "role": "assistant",
                            "content": "The second time I saw",
                        }
                    }
                ]
            ),
        )

        engine = RemoteInferenceEngine(_get_default_model_params())
        conversation1 = Conversation(
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
        conversation2 = Conversation(
            messages=[
                Message(
                    content="Goodbye world!",
                    role=Role.USER,
                ),
                Message(
                    content="Goodbye again!",
                    role=Role.USER,
                ),
            ],
            metadata={"bar": "foo"},
            conversation_id="321",
        )
        expected_result = [
            Conversation(
                messages=[
                    *conversation1.messages,
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
                    *conversation2.messages,
                    Message(
                        content="The second time I saw",
                        role=Role.ASSISTANT,
                    ),
                ],
                metadata={"bar": "foo"},
                conversation_id="321",
            ),
        ]
        result = engine.infer_online(
            [conversation1, conversation2],
            GenerationParams(
                max_new_tokens=5, remote_params=RemoteParams(api_url=_TARGET_SERVER)
            ),
        )
        assert expected_result == result


def test_infer_online_multiple_requests_politeness():
    with aioresponses() as m:
        m.post(
            _TARGET_SERVER,
            status=200,
            payload=dict(
                choices=[
                    {
                        "message": {
                            "role": "assistant",
                            "content": "The first time I saw",
                        }
                    }
                ]
            ),
        )
        m.post(
            _TARGET_SERVER,
            status=200,
            payload=dict(
                choices=[
                    {
                        "message": {
                            "role": "assistant",
                            "content": "The second time I saw",
                        }
                    }
                ]
            ),
        )

        engine = RemoteInferenceEngine(_get_default_model_params())
        conversation1 = Conversation(
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
        conversation2 = Conversation(
            messages=[
                Message(
                    content="Goodbye world!",
                    role=Role.USER,
                ),
                Message(
                    content="Goodbye again!",
                    role=Role.USER,
                ),
            ],
            metadata={"bar": "foo"},
            conversation_id="321",
        )
        expected_result = [
            Conversation(
                messages=[
                    *conversation1.messages,
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
                    *conversation2.messages,
                    Message(
                        content="The second time I saw",
                        role=Role.ASSISTANT,
                    ),
                ],
                metadata={"bar": "foo"},
                conversation_id="321",
            ),
        ]
        start = time.time()
        result = engine.infer_online(
            [conversation1, conversation2],
            GenerationParams(
                max_new_tokens=5,
                remote_params=RemoteParams(
                    api_url=_TARGET_SERVER, politeness_policy=0.5
                ),
            ),
        )
        total_time = time.time() - start
        assert 1.0 < total_time < 1.5
        assert expected_result == result


def test_infer_online_multiple_requests_politeness_multiple_workers():
    with aioresponses() as m:
        m.post(
            _TARGET_SERVER,
            status=200,
            payload=dict(
                choices=[
                    {
                        "message": {
                            "role": "assistant",
                            "content": "The first time I saw",
                        }
                    }
                ]
            ),
        )
        m.post(
            _TARGET_SERVER,
            status=200,
            payload=dict(
                choices=[
                    {
                        "message": {
                            "role": "assistant",
                            "content": "The second time I saw",
                        }
                    }
                ]
            ),
        )

        engine = RemoteInferenceEngine(_get_default_model_params())
        conversation1 = Conversation(
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
        conversation2 = Conversation(
            messages=[
                Message(
                    content="Goodbye world!",
                    role=Role.USER,
                ),
                Message(
                    content="Goodbye again!",
                    role=Role.USER,
                ),
            ],
            metadata={"bar": "foo"},
            conversation_id="321",
        )
        expected_result = [
            Conversation(
                messages=[
                    *conversation1.messages,
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
                    *conversation2.messages,
                    Message(
                        content="The second time I saw",
                        role=Role.ASSISTANT,
                    ),
                ],
                metadata={"bar": "foo"},
                conversation_id="321",
            ),
        ]
        start = time.time()
        result = engine.infer_online(
            [conversation1, conversation2],
            GenerationParams(
                max_new_tokens=5,
                remote_params=RemoteParams(
                    api_url=_TARGET_SERVER,
                    politeness_policy=0.5,
                    num_workers=2,
                ),
            ),
        )
        total_time = time.time() - start
        assert 0.5 < total_time < 1.0
        assert expected_result == result


def test_infer_from_file_empty():
    with tempfile.TemporaryDirectory() as output_temp_dir:
        input_path = Path(output_temp_dir) / "foo" / "input.jsonl"
        _setup_input_conversations(str(input_path), [])
        engine = RemoteInferenceEngine(_get_default_model_params())
        output_path = Path(output_temp_dir) / "b" / "output.jsonl"
        result = engine.infer_online(
            [],
            GenerationParams(
                max_new_tokens=5,
                input_filepath=str(input_path),
                remote_params=RemoteParams(api_url=_TARGET_SERVER, num_workers=2),
                output_filepath=str(output_path),
            ),
        )
        assert [] == result
        infer_result = engine.infer(
            generation_params=GenerationParams(
                max_new_tokens=5,
                input_filepath=str(input_path),
                remote_params=RemoteParams(api_url=_TARGET_SERVER, num_workers=2),
                output_filepath=str(output_path),
            )
        )
        assert [] == infer_result


def test_infer_from_file_to_file():
    with tempfile.TemporaryDirectory() as output_temp_dir:
        input_path = Path(output_temp_dir) / "foo" / "input.jsonl"
        with aioresponses() as m:
            m.post(
                _TARGET_SERVER,
                status=200,
                payload=dict(
                    choices=[
                        {
                            "message": {
                                "role": "assistant",
                                "content": "The first time I saw",
                            }
                        }
                    ]
                ),
            )
            m.post(
                _TARGET_SERVER,
                status=200,
                payload=dict(
                    choices=[
                        {
                            "message": {
                                "role": "assistant",
                                "content": "The second time I saw",
                            }
                        }
                    ]
                ),
            )

            engine = RemoteInferenceEngine(_get_default_model_params())
            conversation1 = Conversation(
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
            conversation2 = Conversation(
                messages=[
                    Message(
                        content="Goodbye world!",
                        role=Role.USER,
                    ),
                    Message(
                        content="Goodbye again!",
                        role=Role.USER,
                    ),
                ],
                metadata={"bar": "foo"},
                conversation_id="321",
            )
            _setup_input_conversations(str(input_path), [conversation1, conversation2])
            expected_result = [
                Conversation(
                    messages=[
                        *conversation1.messages,
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
                        *conversation2.messages,
                        Message(
                            content="The second time I saw",
                            role=Role.ASSISTANT,
                        ),
                    ],
                    metadata={"bar": "foo"},
                    conversation_id="321",
                ),
            ]
            output_path = Path(output_temp_dir) / "b" / "output.jsonl"
            result = engine.infer_online(
                [conversation1, conversation2],
                GenerationParams(
                    max_new_tokens=5,
                    remote_params=RemoteParams(api_url=_TARGET_SERVER, num_workers=2),
                    output_filepath=str(output_path),
                ),
            )
            assert expected_result == result
            # Ensure that intermediary results are saved to the scratch directory.
            with open(output_path.parent / "scratch" / output_path.name) as f:
                parsed_conversations = []
                for line in f:
                    parsed_conversations.append(Conversation.from_json(line))
                assert len(expected_result) == len(parsed_conversations)
            # Ensure the final output is in order.
            with open(output_path) as f:
                parsed_conversations = []
                for line in f:
                    parsed_conversations.append(Conversation.from_json(line))
                assert expected_result == parsed_conversations
