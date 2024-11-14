import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import jsonlines
import pytest
from aioresponses import CallbackResult, aioresponses
from pydantic import BaseModel

from oumi.core.configs import (
    GenerationParams,
    InferenceConfig,
    ModelParams,
    RemoteParams,
)
from oumi.core.configs.params.guided_decoding_params import GuidedDecodingParams
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


def _get_default_inference_config() -> InferenceConfig:
    return InferenceConfig(
        generation=GenerationParams(
            max_new_tokens=5, remote_params=RemoteParams(api_url=_TARGET_SERVER)
        )
    )


def _setup_input_conversations(filepath: str, conversations: list[Conversation]):
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
            _get_default_inference_config(),
        )
        assert expected_result == result


def test_infer_no_remote_params():
    engine = RemoteInferenceEngine(_get_default_model_params())
    with pytest.raises(
        ValueError, match="Remote params must be provided in generation_params."
    ):
        engine.infer_online([], InferenceConfig())
    with pytest.raises(
        ValueError, match="Remote params must be provided in generation_params."
    ):
        engine.infer_from_file("path", InferenceConfig())


def test_infer_online_empty():
    engine = RemoteInferenceEngine(_get_default_model_params())
    expected_result = []
    result = engine.infer_online(
        [],
        _get_default_inference_config(),
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
                _get_default_inference_config(),
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
            _get_default_inference_config(),
        )
        assert expected_result == result


def test_infer_online_multiple_requests():
    # Note: We use the first message's content as the key to avoid
    # stringifying the message object.
    response_by_conversation_id = {
        "Hello world!": dict(
            status=200,
            payload=dict(
                choices=[
                    {
                        "message": {
                            "role": "assistant",
                            "content": "The first time I saw",
                        }
                    }
                ],
            ),
        ),
        "Goodbye world!": dict(
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
        ),
    }

    def response_callback(url, **kwargs):
        request = kwargs.get("json", {})
        conversation_id = request.get("messages", [])[0]["content"][0]["text"]

        if response := response_by_conversation_id.get(conversation_id):
            return CallbackResult(
                status=response["status"],  # type: ignore
                payload=response["payload"],  # type: ignore
            )

        raise ValueError(
            "Test error: Static response not found "
            f"for conversation_id: {conversation_id}"
        )

    with aioresponses() as m:
        m.post(_TARGET_SERVER, callback=response_callback, repeat=True)

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
            _get_default_inference_config(),
        )
        assert expected_result == result


def test_infer_online_multiple_requests_politeness():
    # Note: We use the first message's content as the key to avoid
    # stringifying the message object.
    response_by_conversation_id = {
        "Hello world!": dict(
            status=200,
            payload=dict(
                choices=[
                    {
                        "message": {
                            "role": "assistant",
                            "content": "The first time I saw",
                        }
                    }
                ],
            ),
        ),
        "Goodbye world!": dict(
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
        ),
    }

    def response_callback(url, **kwargs):
        request = kwargs.get("json", {})
        conversation_id = request.get("messages", [])[0]["content"][0]["text"]

        if response := response_by_conversation_id.get(conversation_id):
            return CallbackResult(
                status=response["status"],  # type: ignore
                payload=response["payload"],  # type: ignore
            )

        raise ValueError(
            "Test error: Static response not found "
            f"for conversation_id: {conversation_id}"
        )

    with aioresponses() as m:
        m.post(_TARGET_SERVER, callback=response_callback, repeat=True)

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
        inference_config = InferenceConfig(
            generation=GenerationParams(
                max_new_tokens=5,
                remote_params=RemoteParams(
                    api_url=_TARGET_SERVER, politeness_policy=0.5
                ),
            )
        )
        result = engine.infer_online(
            [conversation1, conversation2],
            inference_config,
        )
        total_time = time.time() - start
        assert 1.0 < total_time < 1.5
        assert expected_result == result


def test_infer_online_multiple_requests_politeness_multiple_workers():
    # Note: We use the first message's content as the key to avoid
    # stringifying the message object.
    response_by_conversation_id = {
        "Hello world!": dict(
            status=200,
            payload=dict(
                choices=[
                    {
                        "message": {
                            "role": "assistant",
                            "content": "The first time I saw",
                        }
                    }
                ],
            ),
        ),
        "Goodbye world!": dict(
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
        ),
    }

    def response_callback(url, **kwargs):
        request = kwargs.get("json", {})
        conversation_id = request.get("messages", [])[0]["content"][0]["text"]

        if response := response_by_conversation_id.get(conversation_id):
            return CallbackResult(
                status=response["status"],  # type: ignore
                payload=response["payload"],  # type: ignore
            )

        raise ValueError(
            "Test error: Static response not found "
            f"for conversation_id: {conversation_id}"
        )

    with aioresponses() as m:
        m.post(_TARGET_SERVER, callback=response_callback, repeat=True)

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
        inference_config = InferenceConfig(
            generation=GenerationParams(
                max_new_tokens=5,
                remote_params=RemoteParams(
                    api_url=_TARGET_SERVER,
                    politeness_policy=0.5,
                    num_workers=2,
                ),
            )
        )
        result = engine.infer_online(
            [conversation1, conversation2],
            inference_config,
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
        inference_config = InferenceConfig(
            input_path=str(input_path),
            output_path=str(output_path),
            generation=GenerationParams(
                max_new_tokens=5,
                remote_params=RemoteParams(api_url=_TARGET_SERVER, num_workers=2),
            ),
        )
        result = engine.infer_online(
            [],
            inference_config,
        )
        assert [] == result
        infer_result = engine.infer(inference_config=inference_config)
        assert [] == infer_result


def test_infer_from_file_to_file():
    with tempfile.TemporaryDirectory() as output_temp_dir:
        input_path = Path(output_temp_dir) / "foo" / "input.jsonl"

        # Note: We use the first message's content as the key to avoid
        # stringifying the message object.
        response_by_conversation_id = {
            "Hello world!": dict(
                status=200,
                payload=dict(
                    choices=[
                        {
                            "message": {
                                "role": "assistant",
                                "content": "The first time I saw",
                            }
                        }
                    ],
                ),
            ),
            "Goodbye world!": dict(
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
            ),
        }

        def response_callback(url, **kwargs):
            request = kwargs.get("json", {})
            conversation_id = request.get("messages", [])[0]["content"][0]["text"]

            if response := response_by_conversation_id.get(conversation_id):
                return CallbackResult(
                    status=response["status"],  # type: ignore
                    payload=response["payload"],  # type: ignore
                )

            raise ValueError(
                "Test error: Static response not found "
                f"for conversation_id: {conversation_id}"
            )

        with aioresponses() as m:
            m.post(_TARGET_SERVER, callback=response_callback, repeat=True)

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
            inference_config = InferenceConfig(
                output_path=str(output_path),
                generation=GenerationParams(
                    max_new_tokens=5,
                    remote_params=RemoteParams(api_url=_TARGET_SERVER, num_workers=2),
                ),
            )
            result = engine.infer_online(
                [conversation1, conversation2],
                inference_config,
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


def test_convert_conversation_to_api_input_with_json_schema():
    """Test conversion with JSON schema guided decoding."""

    class ResponseSchema(BaseModel):
        answer: str
        confidence: float

    engine = RemoteInferenceEngine(_get_default_model_params())
    conversation = Conversation(
        messages=[
            Message(
                content="Hello world!",
                role=Role.USER,
            ),
        ],
        metadata={},
        conversation_id="123",
    )

    generation_params = GenerationParams(
        max_new_tokens=5,
        guided_decoding=GuidedDecodingParams(json=ResponseSchema),
    )

    result = engine._convert_conversation_to_api_input(conversation, generation_params)

    assert result["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "ResponseSchema",
            "schema": ResponseSchema.model_json_schema(),
        },
    }


def test_convert_conversation_to_api_input_without_guided_decoding():
    """Test conversion without guided decoding."""
    engine = RemoteInferenceEngine(_get_default_model_params())
    conversation = Conversation(
        messages=[
            Message(
                content="Hello world!",
                role=Role.USER,
            ),
        ],
        metadata={},
        conversation_id="123",
    )

    generation_params = GenerationParams(max_new_tokens=5)
    result = engine._convert_conversation_to_api_input(conversation, generation_params)

    assert "response_format" not in result


def test_convert_conversation_to_api_input_with_invalid_guided_decoding():
    """Test conversion with invalid guided decoding raises error."""
    engine = RemoteInferenceEngine(_get_default_model_params())
    conversation = Conversation(
        messages=[
            Message(
                content="Hello world!",
                role=Role.USER,
            ),
        ],
        metadata={},
        conversation_id="123",
    )

    generation_params = GenerationParams(
        max_new_tokens=5,
        guided_decoding=GuidedDecodingParams(json=None),
    )

    with pytest.raises(
        ValueError, match="Only JSON schema guided decoding is supported"
    ):
        engine._convert_conversation_to_api_input(conversation, generation_params)


def test_convert_conversation_to_api_input_with_dict_schema():
    """Test conversion with JSON schema provided as a dictionary."""
    engine = RemoteInferenceEngine(_get_default_model_params())
    conversation = Conversation(
        messages=[
            Message(
                content="Hello world!",
                role=Role.USER,
            ),
        ],
        metadata={},
        conversation_id="123",
    )

    schema_dict = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "confidence": {"type": "number"},
        },
        "required": ["answer", "confidence"],
    }

    generation_params = GenerationParams(
        max_new_tokens=5,
        guided_decoding=GuidedDecodingParams(json=schema_dict),
    )

    result = engine._convert_conversation_to_api_input(conversation, generation_params)

    assert result["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "Response",  # Generic name for dict schema
            "schema": schema_dict,
        },
    }


def test_convert_conversation_to_api_input_with_json_string_schema():
    """Test conversion with JSON schema provided as a JSON string."""
    engine = RemoteInferenceEngine(_get_default_model_params())
    conversation = Conversation(
        messages=[
            Message(
                content="Hello world!",
                role=Role.USER,
            ),
        ],
        metadata={},
        conversation_id="123",
    )

    schema_str = """{
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "confidence": {"type": "number"}
        },
        "required": ["answer", "confidence"]
    }"""

    generation_params = GenerationParams(
        max_new_tokens=5,
        guided_decoding=GuidedDecodingParams(json=schema_str),
    )

    result = engine._convert_conversation_to_api_input(conversation, generation_params)

    assert result["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "Response",  # Generic name for string schema
            "schema": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["answer", "confidence"],
            },
        },
    }


def test_convert_conversation_to_api_input_with_invalid_json_string():
    """Test conversion with invalid JSON string raises error."""
    engine = RemoteInferenceEngine(_get_default_model_params())
    conversation = Conversation(
        messages=[
            Message(
                content="Hello world!",
                role=Role.USER,
            ),
        ],
        metadata={},
        conversation_id="123",
    )

    invalid_schema_str = "{invalid json string}"

    generation_params = GenerationParams(
        max_new_tokens=5,
        guided_decoding=GuidedDecodingParams(json=invalid_schema_str),
    )

    with pytest.raises(json.JSONDecodeError):
        engine._convert_conversation_to_api_input(conversation, generation_params)


def test_convert_conversation_to_api_input_with_unsupported_schema_type():
    """Test conversion with unsupported schema type raises error."""
    engine = RemoteInferenceEngine(_get_default_model_params())
    conversation = Conversation(
        messages=[
            Message(
                content="Hello world!",
                role=Role.USER,
            ),
        ],
        metadata={},
        conversation_id="123",
    )

    # Using an integer as an invalid schema type
    invalid_schema = 42

    generation_params = GenerationParams(
        max_new_tokens=5,
        guided_decoding=GuidedDecodingParams(json=invalid_schema),
    )

    with pytest.raises(
        ValueError,
        match="Got unsupported JSON schema type",
    ):
        engine._convert_conversation_to_api_input(conversation, generation_params)


def test_get_request_headers_no_remote_params():
    engine = RemoteInferenceEngine(_get_default_model_params())
    headers = engine._get_request_headers(None)
    assert headers == {}


def test_get_request_headers_with_api_key():
    engine = RemoteInferenceEngine(_get_default_model_params())
    remote_params = RemoteParams(api_url=_TARGET_SERVER, api_key="test-key")
    headers = engine._get_request_headers(remote_params)
    assert headers == {"Authorization": "Bearer test-key"}


def test_get_request_headers_with_env_var():
    with patch.dict(os.environ, {"OPENAI_API_KEY": "env-test-key"}):
        engine = RemoteInferenceEngine(_get_default_model_params())
        remote_params = RemoteParams(
            api_url=_TARGET_SERVER, api_key_env_varname="OPENAI_API_KEY"
        )
        headers = engine._get_request_headers(remote_params)
        assert headers == {"Authorization": "Bearer env-test-key"}


def test_get_request_headers_missing_env_var():
    with patch.dict(os.environ, {}, clear=True):
        engine = RemoteInferenceEngine(_get_default_model_params())
        remote_params = RemoteParams(
            api_url=_TARGET_SERVER, api_key_env_varname="NONEXISTENT_API_KEY"
        )
        headers = engine._get_request_headers(remote_params)
        assert headers == {"Authorization": "Bearer None"}
