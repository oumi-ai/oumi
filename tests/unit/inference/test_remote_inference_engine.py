import tempfile
import time
from pathlib import Path
from typing import Final

import jsonlines
import PIL.Image
import pytest
from aioresponses import CallbackResult, aioresponses

from oumi.core.configs import (
    GenerationParams,
    InferenceConfig,
    ModelParams,
    RemoteParams,
)
from oumi.core.types.conversation import Conversation, Message, Role, Type
from oumi.inference import RemoteInferenceEngine
from oumi.utils.image_utils import (
    base64encode_image_bytes,
    create_png_bytes_from_image,
)
from oumi.utils.io_utils import get_oumi_root_directory

_TARGET_SERVER: Final[str] = "http://fakeurl"
_TEST_IMAGE_DIR: Final[Path] = (
    get_oumi_root_directory().parent.parent.resolve() / "tests" / "testdata" / "images"
)


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


def create_test_text_only_conversation():
    return Conversation(
        messages=[
            Message(content="You are an assistant!", role=Role.SYSTEM, type=Type.TEXT),
            Message(content="Hello", role=Role.USER, type=Type.TEXT),
            Message(content="Hi there!", role=Role.ASSISTANT, type=Type.TEXT),
            Message(content="How are you?", role=Role.USER, type=Type.TEXT),
        ]
    )


def create_test_multimodal_text_image_conversation():
    pil_image = PIL.Image.new(mode="RGB", size=(32, 48))
    png_bytes = create_png_bytes_from_image(pil_image)
    return Conversation(
        messages=[
            Message(content="You are an assistant!", role=Role.SYSTEM, type=Type.TEXT),
            Message(binary=png_bytes, role=Role.USER, type=Type.IMAGE_BINARY),
            Message(content="Hello", role=Role.USER, type=Type.TEXT),
            Message(content="there", role=Role.USER, type=Type.TEXT),
            Message(content="Greetings!", role=Role.ASSISTANT, type=Type.TEXT),
            Message(
                binary=png_bytes,
                content="http://oumi.ai/test.png",
                role=Role.ASSISTANT,
                type=Type.IMAGE_URL,
            ),
            Message(content="Describe this image", role=Role.USER, type=Type.TEXT),
            Message(
                content=str(_TEST_IMAGE_DIR / "the_great_wave_off_kanagawa.jpg"),
                role=Role.USER,
                type=Type.IMAGE_PATH,
            ),
        ]
    )


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


def test_get_list_of_message_json_dicts_multimodal():
    conversation = create_test_multimodal_text_image_conversation()
    assert len(conversation.messages) == 8
    assert conversation[1].type == Type.IMAGE_BINARY and conversation[1].binary
    expected_base64_str = base64encode_image_bytes(
        conversation[1], add_mime_prefix=True
    )
    assert expected_base64_str.startswith("data:image/png;base64,")

    result = RemoteInferenceEngine._get_list_of_message_json_dicts(
        conversation.messages, group_adjacent_same_role_turns=True
    )

    assert len(result) == 4
    assert [m["role"] for m in result] == ["system", "user", "assistant", "user"]

    assert result[0] == {"role": "system", "content": "You are an assistant!"}

    assert result[1]["role"] == "user"
    assert isinstance(result[1]["content"], list) and len(result[1]["content"]) == 3
    assert all([isinstance(item, dict) for item in result[1]["content"]])
    assert result[1]["content"][0] == {
        "type": "image_url",
        "image_url": {"url": expected_base64_str},
    }
    assert result[1]["content"][1] == {"type": "text", "text": "Hello"}
    assert result[1]["content"][2] == {"type": "text", "text": "there"}

    assert result[2]["role"] == "assistant"
    assert isinstance(result[2]["content"], list) and len(result[2]["content"]) == 2
    assert all([isinstance(item, dict) for item in result[2]["content"]])
    assert result[2]["content"][0] == {"type": "text", "text": "Greetings!"}
    assert result[2]["content"][1] == {
        "type": "image_url",
        "image_url": {"url": expected_base64_str},
    }

    assert result[3]["role"] == "user"
    assert isinstance(result[3]["content"], list) and len(result[3]["content"]) == 2
    assert all([isinstance(item, dict) for item in result[3]["content"]])
    assert result[3]["content"][0] == {"type": "text", "text": "Describe this image"}
    tsunami_base64_image_str = result[3]["content"][1]["image_url"]["url"]
    assert isinstance(tsunami_base64_image_str, str)
    assert tsunami_base64_image_str.startswith("data:image/png;base64,")
    assert result[3]["content"][1] == {
        "type": "image_url",
        "image_url": {"url": tsunami_base64_image_str},
    }
