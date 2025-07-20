import asyncio
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Final
from unittest.mock import AsyncMock, patch

import aiohttp
import jsonlines
import PIL.Image
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
from oumi.core.types.conversation import (
    ContentItem,
    Conversation,
    Message,
    Role,
    Type,
)
from oumi.inference import RemoteInferenceEngine
from oumi.inference.remote_inference_engine import BatchStatus
from oumi.utils.conversation_utils import (
    base64encode_content_item_image_bytes,
)
from oumi.utils.image_utils import (
    create_png_bytes_from_image,
)
from tests import get_testdata_dir

_TARGET_SERVER_BASE: Final[str] = "http://fakeurl"
_TARGET_SERVER: Final[str] = "http://fakeurl/somepath"
_TEST_IMAGE_DIR: Final[Path] = get_testdata_dir() / "images"


#
# Fixtures
#
@pytest.fixture
def mock_aioresponse():
    with aioresponses() as m:
        yield m


@pytest.fixture
def mock_asyncio_sleep():
    async def mock_sleep(delay):
        pass

    with patch("asyncio.sleep", side_effect=mock_sleep) as asyncio_sleep:
        yield asyncio_sleep


def _get_default_model_params() -> ModelParams:
    return ModelParams(
        model_name="MlpEncoder",
        trust_remote_code=True,
        tokenizer_name="gpt2",
    )


def _get_default_inference_config() -> InferenceConfig:
    return InferenceConfig(
        generation=GenerationParams(max_new_tokens=5),
        remote_params=RemoteParams(api_url=_TARGET_SERVER),
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
            Message(content="You are an assistant!", role=Role.SYSTEM),
            Message(content="Hello", role=Role.USER),
            Message(content="Hi there!", role=Role.ASSISTANT),
            Message(content="How are you?", role=Role.USER),
        ]
    )


def create_test_png_image_bytes() -> bytes:
    pil_image = PIL.Image.new(mode="RGB", size=(32, 48))
    return create_png_bytes_from_image(pil_image)


def create_test_png_image_base64_str() -> str:
    return base64encode_content_item_image_bytes(
        ContentItem(binary=create_test_png_image_bytes(), type=Type.IMAGE_BINARY),
        add_mime_prefix=True,
    )


def create_test_multimodal_text_image_conversation():
    png_bytes = create_test_png_image_bytes()
    return Conversation(
        messages=[
            Message(content="You are an assistant!", role=Role.SYSTEM),
            Message(
                role=Role.USER,
                content=[
                    ContentItem(binary=png_bytes, type=Type.IMAGE_BINARY),
                    ContentItem(content="Hello", type=Type.TEXT),
                    ContentItem(content="there", type=Type.TEXT),
                ],
            ),
            Message(
                role=Role.ASSISTANT,
                content=[
                    ContentItem(content="Greetings!", type=Type.TEXT),
                    ContentItem(
                        content="http://oumi.ai/test.png",
                        type=Type.IMAGE_URL,
                    ),
                ],
            ),
            Message(
                role=Role.USER,
                content=[
                    ContentItem(content="Describe this image", type=Type.TEXT),
                    ContentItem(
                        content=str(
                            _TEST_IMAGE_DIR / "the_great_wave_off_kanagawa.jpg"
                        ),
                        type=Type.IMAGE_PATH,
                    ),
                ],
            ),
        ]
    )


class AsyncContextManagerMock:
    """Mock for async context manager."""

    def __init__(self):
        self.record_success = AsyncMock()
        self.record_error = AsyncMock()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def read(self):
        return b"test data"


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

        engine = RemoteInferenceEngine(
            model_params=_get_default_model_params(),
            remote_params=RemoteParams(api_url=_TARGET_SERVER),
        )
        conversation = Conversation(
            messages=[
                Message(
                    role=Role.USER,
                    content=[
                        ContentItem(
                            content="Hello world!",
                            type=Type.TEXT,
                        ),
                        ContentItem(
                            content="/tmp/hello/again.png",
                            binary=b"a binary image",
                            type=Type.IMAGE_PATH,
                        ),
                        ContentItem(
                            content="a url for our image",
                            type=Type.IMAGE_URL,
                        ),
                        ContentItem(
                            binary=b"a binary image",
                            type=Type.IMAGE_BINARY,
                        ),
                    ],
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
        result = engine.infer(
            [conversation],
            _get_default_inference_config(),
        )
        assert expected_result == result


def test_infer_online_no_base_api():
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

        engine = RemoteInferenceEngine(
            model_params=_get_default_model_params(),
        )
        conversation = Conversation(
            messages=[
                Message(
                    role=Role.USER,
                    content=[
                        ContentItem(
                            content="Hello world!",
                            type=Type.TEXT,
                        ),
                        ContentItem(
                            content="/tmp/hello/again.png",
                            binary=b"a binary image",
                            type=Type.IMAGE_PATH,
                        ),
                        ContentItem(
                            content="a url for our image",
                            type=Type.IMAGE_URL,
                        ),
                        ContentItem(
                            binary=b"a binary image",
                            type=Type.IMAGE_BINARY,
                        ),
                    ],
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
        result = engine.infer(
            [conversation],
            _get_default_inference_config(),
        )
        assert expected_result == result


def test_infer_online_falls_back_to_default_url():
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

        engine = RemoteInferenceEngine(
            model_params=_get_default_model_params(),
            remote_params=RemoteParams(api_url=_TARGET_SERVER),
        )
        conversation = Conversation(
            messages=[
                Message(
                    role=Role.USER,
                    content=[
                        ContentItem(
                            content="Hello world!",
                            type=Type.TEXT,
                        ),
                        ContentItem(
                            content="/tmp/hello/again.png",
                            binary=b"a binary image",
                            type=Type.IMAGE_PATH,
                        ),
                        ContentItem(
                            content="a url for our image",
                            type=Type.IMAGE_URL,
                        ),
                        ContentItem(
                            binary=b"a binary image",
                            type=Type.IMAGE_BINARY,
                        ),
                    ],
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
        inference_config = _get_default_inference_config()
        inference_config.remote_params = None
        result = engine.infer(
            [conversation],
            inference_config,
        )
        assert expected_result == result


def test_infer_online_falls_back_to_default_api_key():
    def callback(url, **kwargs):
        # Verify our headers
        assert kwargs["headers"]["Authorization"] == "Bearer 1234"

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
            callback=callback,
        )

        engine = RemoteInferenceEngine(
            model_params=_get_default_model_params(),
            remote_params=RemoteParams(api_url=_TARGET_SERVER, api_key="1234"),
        )
        conversation = Conversation(
            messages=[
                Message(
                    role=Role.USER,
                    content=[
                        ContentItem(
                            content="Hello world!",
                            type=Type.TEXT,
                        ),
                        ContentItem(
                            content="/tmp/hello/again.png",
                            binary=b"a binary image",
                            type=Type.IMAGE_PATH,
                        ),
                        ContentItem(
                            content="a url for our image",
                            type=Type.IMAGE_URL,
                        ),
                        ContentItem(
                            binary=b"a binary image",
                            type=Type.IMAGE_BINARY,
                        ),
                    ],
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
        inference_config = _get_default_inference_config()
        inference_config.remote_params = None
        result = engine.infer(
            [conversation],
            inference_config,
        )
        assert expected_result == result


def test_infer_online_falls_back_to_default_api_key_env_varname(monkeypatch):
    def callback(url, **kwargs):
        # Verify our headers
        assert kwargs["headers"]["Authorization"] == "Bearer 4321"

    monkeypatch.setenv("NEW_API_KEY_VAR", "4321")
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
            callback=callback,
        )

        engine = RemoteInferenceEngine(
            model_params=_get_default_model_params(),
            remote_params=RemoteParams(
                api_url=_TARGET_SERVER, api_key_env_varname="NEW_API_KEY_VAR"
            ),
        )
        conversation = Conversation(
            messages=[
                Message(
                    role=Role.USER,
                    content=[
                        ContentItem(
                            content="Hello world!",
                            type=Type.TEXT,
                        ),
                        ContentItem(
                            content="/tmp/hello/again.png",
                            binary=b"a binary image",
                            type=Type.IMAGE_PATH,
                        ),
                        ContentItem(
                            content="a url for our image",
                            type=Type.IMAGE_URL,
                        ),
                        ContentItem(
                            binary=b"a binary image",
                            type=Type.IMAGE_BINARY,
                        ),
                    ],
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
        inference_config = _get_default_inference_config()
        inference_config.remote_params = None
        result = engine.infer(
            [conversation],
            inference_config,
        )
        assert expected_result == result


def test_infer_no_remote_params_api_url():
    with pytest.raises(ValueError, match="API URL is required for remote inference."):
        engine = RemoteInferenceEngine(
            model_params=_get_default_model_params(),
        )
        engine.infer(
            input=[Conversation(messages=[])],
        )


def test_infer_no_api_key():
    with pytest.raises(
        ValueError,
        match=(
            r"An API key is required for remote inference with the "
            r"`RemoteInferenceEngine` inference engine. Please set the environment "
            r"variable `MY_API_KEY`."
        ),
    ):
        engine = RemoteInferenceEngine(
            model_params=_get_default_model_params(),
            remote_params=RemoteParams(
                api_url=_TARGET_SERVER,
                api_key_env_varname="MY_API_KEY",  # Indicates that API key is required.
            ),
        )
        engine.infer(
            input=[Conversation(messages=[])],
        )


def test_infer_online_empty():
    engine = RemoteInferenceEngine(
        _get_default_model_params(), remote_params=RemoteParams(api_url=_TARGET_SERVER)
    )
    expected_result = []
    result = engine.infer(
        [],
        _get_default_inference_config(),
    )
    assert expected_result == result


def test_infer_online_fast_fail_nonretriable(mock_asyncio_sleep):
    with aioresponses() as m:
        # Only set up one response since it should fail immediately
        m.post(
            _TARGET_SERVER, status=401, payload={"error": {"message": "Unauthorized"}}
        )

        engine = RemoteInferenceEngine(
            _get_default_model_params(),
            remote_params=RemoteParams(
                api_url=_TARGET_SERVER,
                max_retries=3,  # Even with retries configured, should fail immediately
            ),
        )
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
        with pytest.raises(RuntimeError, match="Non-retriable error: Unauthorized"):
            _ = engine.infer(
                [conversation],
                _get_default_inference_config(),
            )
        # Should not retry on 401
        assert mock_asyncio_sleep.call_count == 0


def test_infer_online_fails_with_message(mock_asyncio_sleep):
    with aioresponses() as m:
        m.post(
            _TARGET_SERVER,
            status=504,
            payload={"error": {"message": "Gateway timeout"}},
        )
        m.post(
            _TARGET_SERVER,
            status=429,
            payload={"error": {"message": "Too many requests"}},
        )
        m.post(
            _TARGET_SERVER,
            status=503,
            payload={"error": {"message": "Service unavailable"}},
        )
        m.post(
            _TARGET_SERVER,
            status=500,
            payload={"error": {"message": "Internal server error"}},
        )

        engine = RemoteInferenceEngine(
            _get_default_model_params(),
            remote_params=RemoteParams(api_url=_TARGET_SERVER, max_retries=0),
        )
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
        config = _get_default_inference_config()
        if config.remote_params is not None:
            config.remote_params.max_retries = 0

        with pytest.raises(
            RuntimeError,
            match="Failed to query API after 1 attempts. Reason: Gateway timeout",
        ):
            _ = engine.infer(
                [conversation],
                config,
            )
        with pytest.raises(
            RuntimeError,
            match="Failed to query API after 1 attempts. Reason: Too many requests",
        ):
            _ = engine.infer(
                [conversation],
                config,
            )
        with pytest.raises(
            RuntimeError,
            match="Failed to query API after 1 attempts. Reason: Service unavailable",
        ):
            _ = engine.infer(
                [conversation],
                config,
            )
        with pytest.raises(
            RuntimeError,
            match="Failed to query API after 1 attempts. Reason: Internal server error",
        ):
            _ = engine.infer(
                [conversation],
                config,
            )

        # No retries
        assert mock_asyncio_sleep.call_count == 0


def test_infer_online_fails_with_message_and_retries(mock_asyncio_sleep):
    with aioresponses() as m:
        m.post(
            _TARGET_SERVER,
            status=500,
            payload={"error": {"message": "Internal server error"}},
        )
        m.post(
            _TARGET_SERVER,
            status=500,
            payload={"error": {"message": "Internal server error"}},
        )
        m.post(
            _TARGET_SERVER,
            status=500,
            payload={"error": {"message": "Internal server error"}},
        )
        m.post(
            _TARGET_SERVER,
            status=500,
            payload={"error": {"message": "Internal server error"}},
        )

        config = _get_default_inference_config()
        engine = RemoteInferenceEngine(
            _get_default_model_params(),
            remote_params=RemoteParams(api_url=_TARGET_SERVER),
        )
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
        with pytest.raises(
            RuntimeError,
            match="Failed to query API after 4 attempts. Reason: Internal server error",
        ):
            _ = engine.infer(
                [conversation],
                config,
            )
        # 3 retries + 3 backoffs
        assert mock_asyncio_sleep.call_count == 6


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

        engine = RemoteInferenceEngine(
            _get_default_model_params(),
            remote_params=RemoteParams(api_url=_TARGET_SERVER),
        )
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
        result = engine.infer(
            [conversation],
            _get_default_inference_config(),
        )
        assert expected_result == result


def test_infer_online_multiple_requests():
    # Note: We use the first message's content as the key to avoid
    # stringifying the message object.
    response_by_conversation_id = {
        "Hello world!": {
            "status": 200,
            "payload": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "The first time I saw",
                        }
                    }
                ]
            },
        },
        "Goodbye world!": {
            "status": 200,
            "payload": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "The second time I saw",
                        }
                    }
                ]
            },
        },
    }

    def response_callback(url: str, **kwargs: Any) -> CallbackResult:
        """Callback for mocked API responses."""
        request = kwargs.get("json", {})
        messages = request.get("messages", [])
        if not messages:
            raise ValueError("No messages in request")
        content = messages[0].get("content", [])
        if not content:
            raise ValueError("No content in message")
        conversation_id = content[0].get("text")

        if response := response_by_conversation_id.get(conversation_id):
            # Extract status and payload from the response dict
            return CallbackResult(
                status=response["status"], payload=response["payload"]
            )

        raise ValueError(
            "Test error: Static response not found for "
            f"conversation_id: {conversation_id}"
        )

    with aioresponses() as m:
        m.post(_TARGET_SERVER, callback=response_callback, repeat=True)

        engine = RemoteInferenceEngine(
            _get_default_model_params(),
            remote_params=RemoteParams(api_url=_TARGET_SERVER),
        )
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
        result = engine.infer(
            [conversation1, conversation2],
            _get_default_inference_config(),
        )
        assert expected_result == result


def test_infer_online_multiple_requests_politeness():
    # Note: We use the first message's content as the key to avoid
    # stringifying the message object.
    response_by_conversation_id = {
        "Hello world!": {
            "status": 200,
            "payload": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "The first time I saw",
                        }
                    }
                ]
            },
        },
        "Goodbye world!": {
            "status": 200,
            "payload": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "The second time I saw",
                        }
                    }
                ]
            },
        },
    }

    def response_callback(url: str, **kwargs: Any) -> CallbackResult:
        """Callback for mocked API responses."""
        request = kwargs.get("json", {})
        messages = request.get("messages", [])
        if not messages:
            raise ValueError("No messages in request")
        content = messages[0].get("content", [])
        if not content:
            raise ValueError("No content in message")
        conversation_id = content[0].get("text")

        if response := response_by_conversation_id.get(conversation_id):
            # Extract status and payload from the response dict
            return CallbackResult(
                status=response["status"], payload=response["payload"]
            )

        raise ValueError(
            "Test error: Static response not found for "
            f"conversation_id: {conversation_id}"
        )

    with aioresponses() as m:
        m.post(_TARGET_SERVER, callback=response_callback, repeat=True)

        remote_params = RemoteParams(api_url=_TARGET_SERVER, politeness_policy=0.5)
        engine = RemoteInferenceEngine(
            _get_default_model_params(), remote_params=remote_params
        )
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
            ),
            remote_params=remote_params,
        )
        result = engine.infer(
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
        "Hello world!": {
            "status": 200,
            "payload": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "The first time I saw",
                        }
                    }
                ]
            },
        },
        "Goodbye world!": {
            "status": 200,
            "payload": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "The second time I saw",
                        }
                    }
                ]
            },
        },
    }

    def response_callback(url: str, **kwargs: Any) -> CallbackResult:
        """Callback for mocked API responses."""
        request = kwargs.get("json", {})
        messages = request.get("messages", [])
        if not messages:
            raise ValueError("No messages in request")
        content = messages[0].get("content", [])
        if not content:
            raise ValueError("No content in message")
        conversation_id = content[0].get("text")

        if response := response_by_conversation_id.get(conversation_id):
            # Extract status and payload from the response dict
            return CallbackResult(
                status=response["status"], payload=response["payload"]
            )

        raise ValueError(
            "Test error: Static response not found "
            f"for conversation_id: {conversation_id}"
        )

    with aioresponses() as m:
        m.post(_TARGET_SERVER, callback=response_callback, repeat=True)

        remote_params = RemoteParams(
            api_url=_TARGET_SERVER,
            politeness_policy=0.5,
            num_workers=2,
            use_adaptive_concurrency=False,
        )
        engine = RemoteInferenceEngine(
            _get_default_model_params(), remote_params=remote_params
        )

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
            ),
            remote_params=remote_params,
        )
        result = engine.infer(
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
        remote_params = RemoteParams(api_url=_TARGET_SERVER, num_workers=2)
        engine = RemoteInferenceEngine(
            _get_default_model_params(), remote_params=remote_params
        )
        output_path = Path(output_temp_dir) / "b" / "output.jsonl"
        inference_config = InferenceConfig(
            input_path=str(input_path),
            output_path=str(output_path),
            generation=GenerationParams(
                max_new_tokens=5,
            ),
            remote_params=remote_params,
        )
        infer_result = engine.infer(inference_config=inference_config)
        assert [] == infer_result


def test_infer_from_file_to_file():
    with tempfile.TemporaryDirectory() as output_temp_dir:
        input_path = Path(output_temp_dir) / "foo" / "input.jsonl"

        # Note: We use the first message's content as the key to avoid
        # stringifying the message object.
        response_by_conversation_id = {
            "Hello world!": {
                "status": 200,
                "payload": {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "The first time I saw",
                            }
                        }
                    ]
                },
            },
            "Goodbye world!": {
                "status": 200,
                "payload": {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "The second time I saw",
                            }
                        }
                    ]
                },
            },
        }

        def response_callback(url: str, **kwargs: Any) -> CallbackResult:
            """Callback for mocked API responses."""
            request = kwargs.get("json", {})
            messages = request.get("messages", [])
            if not messages:
                raise ValueError("No messages in request")
            content = messages[0].get("content", [])
            if not content:
                raise ValueError("No content in message")
            conversation_id = content[0].get("text")

            if response := response_by_conversation_id.get(conversation_id):
                # Extract status and payload from the response dict
                return CallbackResult(
                    status=response["status"], payload=response["payload"]
                )

            raise ValueError(
                "Test error: Static response not found for "
                f"conversation_id: {conversation_id}"
            )

        with aioresponses() as m:
            m.post(_TARGET_SERVER, callback=response_callback, repeat=True)
            remote_params = RemoteParams(api_url=_TARGET_SERVER, num_workers=2)
            engine = RemoteInferenceEngine(
                _get_default_model_params(), remote_params=remote_params
            )
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
                ),
                remote_params=remote_params,
            )
            result = engine.infer(
                [conversation1, conversation2],
                inference_config,
            )
            assert expected_result == result
            # Ensure that intermediary results are cleaned up after inference.
            scratch_path = output_path.parent / "scratch" / output_path.name
            assert not scratch_path.exists(), f"Scratch file {scratch_path} exists"
            # Ensure the final output is in order.
            with open(output_path) as f:
                parsed_conversations = []
                for line in f:
                    parsed_conversations.append(Conversation.from_json(line))
                assert expected_result == parsed_conversations


def test_infer_from_file_to_file_failure_midway():
    with tempfile.TemporaryDirectory() as output_temp_dir:
        input_path = Path(output_temp_dir) / "foo" / "input.jsonl"

        # Note: We use the first message's content as the key to avoid
        # stringifying the message object.
        response_by_conversation_id = {
            "Hello world!": {
                "status": 200,
                "payload": {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "The first time I saw",
                            }
                        }
                    ]
                },
            },
            "Goodbye world!": {
                "status": 500,  # Fail the second conversation
                "payload": {"error": {"message": "Internal server error"}},
            },
        }

        def response_callback(url: str, **kwargs: Any) -> CallbackResult:
            """Callback for mocked API responses."""
            request = kwargs.get("json", {})
            messages = request.get("messages", [])
            if not messages:
                raise ValueError("No messages in request")
            content = messages[0].get("content", [])
            if not content:
                raise ValueError("No content in message")
            conversation_id = content[0].get("text")

            if response := response_by_conversation_id.get(conversation_id):
                # Extract status and payload from the response dict
                return CallbackResult(
                    status=response["status"], payload=response["payload"]
                )

            raise ValueError(
                "Test error: Static response not found for "
                f"conversation_id: {conversation_id}"
            )

        with aioresponses() as m:
            m.post(_TARGET_SERVER, callback=response_callback, repeat=True)
            remote_params = RemoteParams(api_url=_TARGET_SERVER, num_workers=2)
            engine = RemoteInferenceEngine(
                _get_default_model_params(), remote_params=remote_params
            )
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
                )
            ]
            output_path = Path(output_temp_dir) / "b" / "output.jsonl"
            inference_config = InferenceConfig(
                output_path=str(output_path),
                generation=GenerationParams(
                    max_new_tokens=5,
                ),
                remote_params=remote_params,
            )

            with pytest.raises(RuntimeError, match="Internal server error"):
                _ = engine.infer(
                    [conversation1, conversation2],
                    inference_config,
                )

            # Verify scratch file exists and contains only first conversation
            scratch_path = output_path.parent / "scratch" / output_path.name
            assert scratch_path.exists(), f"Scratch file {scratch_path} does not exist"

            # Read scratch file and verify contents
            with open(scratch_path) as f:
                parsed_conversations = []
                for line in f:
                    parsed_conversations.append(Conversation.from_json(line))
                assert expected_result == parsed_conversations

            # Ensure the final output is in order and matches scratch file
            assert not output_path.exists(), f"Output file {output_path} exists"


def test_get_list_of_message_json_dicts_multimodal_with_grouping():
    conversation = create_test_multimodal_text_image_conversation()
    assert len(conversation.messages) == 4
    expected_base64_str = create_test_png_image_base64_str()
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
        "image_url": {"url": "http://oumi.ai/test.png"},
    }

    assert result[3]["role"] == "user"
    assert isinstance(result[3]["content"], list) and len(result[3]["content"]) == 2
    assert all([isinstance(item, dict) for item in result[3]["content"]])
    assert result[3]["content"][0] == {"type": "text", "text": "Describe this image"}
    content = result[3]["content"][1]
    assert isinstance(content, dict)
    assert "image_url" in content
    image_url = content["image_url"]
    assert isinstance(image_url, dict)
    assert "url" in image_url
    tsunami_base64_image_str = image_url["url"]

    assert isinstance(tsunami_base64_image_str, str)
    assert tsunami_base64_image_str.startswith("data:image/png;base64,")
    assert content == {
        "type": "image_url",
        "image_url": {"url": tsunami_base64_image_str},
    }


@pytest.mark.parametrize(
    "conversation,group_adjacent_same_role_turns",
    [
        (create_test_multimodal_text_image_conversation(), False),
        (create_test_text_only_conversation(), False),
        (create_test_text_only_conversation(), True),
    ],
)
def test_get_list_of_message_json_dicts_multimodal_no_grouping(
    conversation: Conversation, group_adjacent_same_role_turns: bool
):
    result = RemoteInferenceEngine._get_list_of_message_json_dicts(
        conversation.messages,
        group_adjacent_same_role_turns=group_adjacent_same_role_turns,
    )

    assert len(result) == len(conversation.messages)
    assert [m["role"] for m in result] == [m.role for m in conversation.messages]

    for i in range(len(result)):
        json_dict = result[i]
        message = conversation.messages[i]
        debug_info = f"Index: {i} JSON: {json_dict} Message: {message}"
        if len(debug_info) > 1024:
            debug_info = debug_info[:1024] + " ..."

        assert "role" in json_dict, debug_info
        assert message.role == json_dict["role"], debug_info
        if isinstance(message.content, str):
            assert isinstance(json_dict["content"], str), debug_info
            assert message.content == json_dict["content"], debug_info
        else:
            assert isinstance(message.content, list), debug_info
            assert "content" in json_dict, debug_info
            assert isinstance(json_dict["content"], list), debug_info
            assert len(message.content) == len(json_dict["content"]), debug_info

            assert message.contains_images(), debug_info

            for idx, item in enumerate(message.content):
                json_item = json_dict["content"][idx]
                assert isinstance(json_item, dict)
                assert "type" in json_item, debug_info

                if item.is_text():
                    assert json_item["type"] == "text", debug_info
                    assert json_item["text"] == item.content, debug_info
                elif item.is_image():
                    assert json_item["type"] == "image_url", debug_info
                    assert "image_url" in json_item, debug_info
                    assert isinstance(json_item["image_url"], dict), debug_info
                    assert "url" in json_item["image_url"], debug_info
                    assert isinstance(json_item["image_url"]["url"], str), debug_info
                    if item.type == Type.IMAGE_BINARY:
                        assert "image_url" in json_item
                        image_url = json_item["image_url"]
                        assert isinstance(image_url, dict)
                        assert "url" in image_url
                        expected_base64_bytes_str = (
                            base64encode_content_item_image_bytes(
                                message.image_content_items[-1], add_mime_prefix=True
                            )
                        )
                        assert len(expected_base64_bytes_str) == len(image_url["url"])
                        assert image_url == {"url": expected_base64_bytes_str}, (
                            debug_info
                        )
                    elif item.type == Type.IMAGE_URL:
                        assert json_item["image_url"] == {"url": item.content}, (
                            debug_info
                        )
                    elif item.type == Type.IMAGE_PATH:
                        assert json_item["image_url"]["url"].startswith(
                            "data:image/png;base64,"
                        ), debug_info


def test_convert_conversation_to_api_input_with_json_schema():
    """Test conversion with JSON schema guided decoding."""

    class ResponseSchema(BaseModel):
        answer: str
        confidence: float

    engine = RemoteInferenceEngine(
        _get_default_model_params(), remote_params=RemoteParams(api_url=_TARGET_SERVER)
    )
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

    result = engine._convert_conversation_to_api_input(
        conversation, generation_params, _get_default_model_params()
    )

    assert result["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "ResponseSchema",
            "schema": ResponseSchema.model_json_schema(),
        },
    }


def test_convert_conversation_to_api_input_without_guided_decoding():
    """Test conversion without guided decoding."""
    engine = RemoteInferenceEngine(
        _get_default_model_params(), remote_params=RemoteParams(api_url=_TARGET_SERVER)
    )
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
    result = engine._convert_conversation_to_api_input(
        conversation, generation_params, _get_default_model_params()
    )

    assert "response_format" not in result


def test_convert_conversation_to_api_input_with_invalid_guided_decoding():
    """Test conversion with invalid guided decoding raises error."""
    engine = RemoteInferenceEngine(
        _get_default_model_params(), remote_params=RemoteParams(api_url=_TARGET_SERVER)
    )
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
        engine._convert_conversation_to_api_input(
            conversation, generation_params, _get_default_model_params()
        )


def test_convert_conversation_to_api_input_with_dict_schema():
    """Test conversion with JSON schema provided as a dictionary."""
    engine = RemoteInferenceEngine(
        _get_default_model_params(), remote_params=RemoteParams(api_url=_TARGET_SERVER)
    )
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

    result = engine._convert_conversation_to_api_input(
        conversation, generation_params, _get_default_model_params()
    )

    assert result["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "Response",  # Generic name for dict schema
            "schema": schema_dict,
        },
    }


def test_convert_conversation_to_api_input_with_json_string_schema():
    """Test conversion with JSON schema provided as a JSON string."""
    engine = RemoteInferenceEngine(
        _get_default_model_params(), remote_params=RemoteParams(api_url=_TARGET_SERVER)
    )
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

    result = engine._convert_conversation_to_api_input(
        conversation, generation_params, _get_default_model_params()
    )

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
    engine = RemoteInferenceEngine(
        _get_default_model_params(), remote_params=RemoteParams(api_url=_TARGET_SERVER)
    )
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
        engine._convert_conversation_to_api_input(
            conversation, generation_params, _get_default_model_params()
        )


def test_convert_conversation_to_api_input_with_unsupported_schema_type():
    """Test conversion with unsupported schema type raises error."""
    engine = RemoteInferenceEngine(
        _get_default_model_params(), remote_params=RemoteParams(api_url=_TARGET_SERVER)
    )
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
        engine._convert_conversation_to_api_input(
            conversation, generation_params, _get_default_model_params()
        )


def test_get_request_headers_no_remote_params():
    engine = RemoteInferenceEngine(
        _get_default_model_params(), remote_params=RemoteParams(api_url=_TARGET_SERVER)
    )
    headers = engine._get_request_headers(None)
    assert headers == {}


def test_get_request_headers_with_api_key():
    remote_params = RemoteParams(api_url=_TARGET_SERVER, api_key="test-key")
    engine = RemoteInferenceEngine(
        _get_default_model_params(), remote_params=remote_params
    )
    headers = engine._get_request_headers(remote_params)
    assert headers == {"Authorization": "Bearer test-key"}


def test_get_request_headers_without_api_key():
    remote_params = RemoteParams(api_url=_TARGET_SERVER)
    engine = RemoteInferenceEngine(
        _get_default_model_params(), remote_params=remote_params
    )
    headers = engine._get_request_headers(remote_params)
    assert headers == {}


def test_get_request_headers_with_env_var():
    with patch.dict(os.environ, {"OPENAI_API_KEY": "env-test-key"}):
        remote_params = RemoteParams(
            api_url=_TARGET_SERVER, api_key_env_varname="OPENAI_API_KEY"
        )
        engine = RemoteInferenceEngine(
            _get_default_model_params(),
            remote_params=remote_params,
        )
        headers = engine._get_request_headers(remote_params)
        assert headers == {"Authorization": "Bearer env-test-key"}


def test_get_request_headers_missing_env_var():
    with patch.dict(os.environ, {}, clear=True):
        remote_params = RemoteParams(
            api_url=_TARGET_SERVER, api_key_env_varname="NONEXISTENT_API_KEY"
        )
        engine = RemoteInferenceEngine(
            _get_default_model_params(),
            remote_params=remote_params,
        )
        headers = engine._get_request_headers(remote_params)
        assert headers == {}


@pytest.mark.asyncio
async def test_upload_batch_file():
    """Test uploading a batch file."""
    with aioresponses() as m:
        m.post(
            f"{_TARGET_SERVER_BASE}/v1/files",
            status=200,
            payload={"id": "file-123"},
        )

        engine = RemoteInferenceEngine(
            _get_default_model_params(),
            remote_params=RemoteParams(api_url=_TARGET_SERVER),
        )

        batch_requests = [
            {
                "custom_id": "request-1",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {"messages": [{"role": "user", "content": "Hello"}]},
            }
        ]

        # Patch aiofiles.open to return our async context manager
        with patch("aiofiles.open", return_value=AsyncContextManagerMock()):
            file_id = await engine._upload_batch_file(batch_requests)
            assert file_id == "file-123"


@pytest.mark.asyncio
async def test_create_batch():
    """Test creating a batch job."""
    with aioresponses() as m:
        # Mock file upload
        m.post(
            f"{_TARGET_SERVER_BASE}/v1/files",
            status=200,
            payload={"id": "file-123"},
        )
        # Mock batch creation
        m.post(
            f"{_TARGET_SERVER_BASE}/v1/batches",
            status=200,
            payload={"id": "batch-456"},
        )

        engine = RemoteInferenceEngine(
            _get_default_model_params(),
            remote_params=RemoteParams(api_url=_TARGET_SERVER),
        )

        conversation = Conversation(
            messages=[
                Message(content="Hello", role=Role.USER),
            ]
        )

        with patch("aiofiles.open", return_value=AsyncContextManagerMock()):
            batch_id = await engine._create_batch(
                [conversation],
                _get_default_inference_config().generation,
                _get_default_model_params(),
            )
            assert batch_id == "batch-456"


@pytest.mark.asyncio
async def test_get_batch_status():
    """Test getting batch status."""
    with aioresponses() as m:
        m.get(
            f"{_TARGET_SERVER_BASE}/v1/batches/batch-123",
            status=200,
            payload={
                "id": "batch-123",
                "status": "completed",
                "request_counts": {
                    "total": 10,
                    "completed": 8,
                    "failed": 1,
                },
            },
        )

        engine = RemoteInferenceEngine(
            _get_default_model_params(),
            remote_params=RemoteParams(api_url=_TARGET_SERVER),
        )

        status = await engine._get_batch_status("batch-123")
        assert status.id == "batch-123"
        assert status.status == BatchStatus.COMPLETED
        assert status.total_requests == 10
        assert status.completed_requests == 8
        assert status.failed_requests == 1


@pytest.mark.asyncio
async def test_get_batch_results():
    """Test getting batch results."""
    with aioresponses() as m:
        # Mock batch status request
        m.get(
            f"{_TARGET_SERVER_BASE}/v1/batches/batch-123",
            status=200,
            payload={
                "id": "batch-123",
                "status": "completed",
                "request_counts": {
                    "total": 1,
                    "completed": 1,
                    "failed": 0,
                },
                "output_file_id": "file-output-123",
            },
        )

        # Mock file content request
        m.get(
            f"{_TARGET_SERVER_BASE}/v1/files/file-output-123/content",
            status=200,
            body=json.dumps(
                {
                    "custom_id": "request-1",
                    "response": {
                        "body": {
                            "choices": [
                                {
                                    "message": {
                                        "role": "assistant",
                                        "content": "Hello there!",
                                    }
                                }
                            ]
                        }
                    },
                }
            ),
        )

        engine = RemoteInferenceEngine(
            _get_default_model_params(),
            remote_params=RemoteParams(api_url=_TARGET_SERVER),
        )

        conversation = Conversation(
            messages=[
                Message(content="Hello", role=Role.USER),
            ]
        )

        results = await engine._get_batch_results_with_mapping(
            "batch-123",
            [conversation],
        )

        assert len(results) == 1
        assert results[0].messages[-1].content == "Hello there!"
        assert results[0].messages[-1].role == Role.ASSISTANT


def test_infer_batch():
    """Test the public infer_batch method."""
    with aioresponses() as m:
        # Mock file upload
        m.post(
            f"{_TARGET_SERVER_BASE}/v1/files",
            status=200,
            payload={"id": "file-123"},
        )
        # Mock batch creation
        m.post(
            f"{_TARGET_SERVER_BASE}/v1/batches",
            status=200,
            payload={"id": "batch-456"},
        )

        engine = RemoteInferenceEngine(
            _get_default_model_params(),
            remote_params=RemoteParams(api_url=_TARGET_SERVER),
        )

        conversation = Conversation(
            messages=[
                Message(content="Hello", role=Role.USER),
            ]
        )

        # Use AsyncContextManagerMock instead of AsyncMock
        with patch("aiofiles.open", return_value=AsyncContextManagerMock()):
            batch_id = engine.infer_batch(
                [conversation], _get_default_inference_config()
            )
            assert batch_id == "batch-456"


def test_get_batch_status_public():
    """Test the public get_batch_status method."""
    with aioresponses() as m:
        m.get(
            f"{_TARGET_SERVER_BASE}/v1/batches/batch-123",
            status=200,
            payload={
                "id": "batch-123",
                "status": "in_progress",
                "request_counts": {
                    "total": 10,
                    "completed": 5,
                    "failed": 0,
                },
            },
        )

        engine = RemoteInferenceEngine(
            _get_default_model_params(),
            remote_params=RemoteParams(api_url=_TARGET_SERVER),
        )

        status = engine.get_batch_status("batch-123")
        assert status.id == "batch-123"
        assert status.status == BatchStatus.IN_PROGRESS
        assert status.total_requests == 10
        assert status.completed_requests == 5
        assert status.failed_requests == 0


def test_get_batch_results_public():
    """Test the public get_batch_results method."""
    with aioresponses() as m:
        # Mock batch status request
        m.get(
            f"{_TARGET_SERVER_BASE}/v1/batches/batch-123",
            status=200,
            payload={
                "id": "batch-123",
                "status": "completed",
                "request_counts": {
                    "total": 1,
                    "completed": 1,
                    "failed": 0,
                },
                "output_file_id": "file-output-123",
            },
        )

        # Mock file content request
        m.get(
            f"{_TARGET_SERVER_BASE}/v1/files/file-output-123/content",
            status=200,
            body=json.dumps(
                {
                    "custom_id": "request-1",
                    "response": {
                        "body": {
                            "choices": [
                                {
                                    "message": {
                                        "role": "assistant",
                                        "content": "Hello there!",
                                    }
                                }
                            ]
                        }
                    },
                }
            ),
        )

        engine = RemoteInferenceEngine(
            _get_default_model_params(),
            remote_params=RemoteParams(api_url=_TARGET_SERVER),
        )

        conversation = Conversation(
            messages=[
                Message(content="Hello", role=Role.USER),
            ]
        )

        results = engine.get_batch_results(
            "batch-123",
            [conversation],
        )

        assert len(results) == 1
        assert results[0].messages[-1].content == "Hello there!"
        assert results[0].messages[-1].role == Role.ASSISTANT


@pytest.mark.asyncio
async def test_list_batches():
    """Test listing batch jobs."""
    with aioresponses() as m:
        # Mock with exact URL including query parameters
        m.get(
            f"{_TARGET_SERVER_BASE}/v1/batches?limit=1",  # Include query params in URL
            status=200,
            payload={
                "object": "list",
                "data": [
                    {
                        "id": "batch_abc123",
                        "object": "batch",
                        "endpoint": "/v1/chat/completions",
                        "input_file_id": "file-abc123",
                        "batch_completion_window": "24h",
                        "status": "completed",
                        "output_file_id": "file-cvaTdG",
                        "error_file_id": "file-HOWS94",
                        "created_at": 1711471533,
                        "request_counts": {"total": 100, "completed": 95, "failed": 5},
                        "metadata": {"batch_description": "Test batch"},
                    }
                ],
                "first_id": "batch_abc123",
                "last_id": "batch_abc123",
                "has_more": False,
            },
        )

        engine = RemoteInferenceEngine(
            _get_default_model_params(),
            remote_params=RemoteParams(api_url=_TARGET_SERVER),
        )

        response = await engine._list_batches(limit=1)

        assert len(response.batches) == 1
        batch = response.batches[0]
        assert batch.id == "batch_abc123"
        assert batch.status == BatchStatus.COMPLETED
        assert batch.total_requests == 100
        assert batch.completed_requests == 95
        assert batch.failed_requests == 5
        assert batch.metadata == {"batch_description": "Test batch"}
        assert response.first_id == "batch_abc123"
        assert response.last_id == "batch_abc123"
        assert not response.has_more


def test_list_batches_public():
    """Test the public list_batches method."""
    with aioresponses() as m:
        m.get(
            f"{_TARGET_SERVER_BASE}/v1/batches?limit=2",  # Include query params in URL
            status=200,
            payload={
                "object": "list",
                "data": [
                    {
                        "id": "batch_1",
                        "endpoint": "/v1/chat/completions",
                        "status": "completed",
                        "input_file_id": "file-1",
                        "batch_completion_window": "24h",
                    },
                    {
                        "id": "batch_2",
                        "endpoint": "/v1/chat/completions",
                        "status": "in_progress",
                        "input_file_id": "file-2",
                        "batch_completion_window": "24h",
                    },
                ],
                "first_id": "batch_1",
                "last_id": "batch_2",
                "has_more": True,
            },
        )

        engine = RemoteInferenceEngine(
            _get_default_model_params(),
            remote_params=RemoteParams(api_url=_TARGET_SERVER),
        )

        response = engine.list_batches(limit=2)

        assert len(response.batches) == 2
        assert response.first_id == "batch_1"
        assert response.last_id == "batch_2"
        assert response.has_more


def test_infer_online_handles_content_type_text_plain():
    """Test that the engine can handle text/plain responses and parse them as JSON."""
    with aioresponses() as m:
        m.post(
            _TARGET_SERVER,
            status=200,
            body=json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "The first time I saw",
                            }
                        }
                    ]
                }
            ),
            content_type="text/plain",
        )

        engine = RemoteInferenceEngine(
            model_params=_get_default_model_params(),
            remote_params=RemoteParams(api_url=_TARGET_SERVER),
        )
        conversation = Conversation(
            messages=[
                Message(
                    role=Role.USER,
                    content="Hello world!",
                ),
            ],
        )
        inference_config = _get_default_inference_config()
        inference_config.remote_params = engine._remote_params
        result = engine.infer(
            [conversation],
            inference_config,
        )
        assert len(result) == 1
        assert result[0].messages[-1].content == "The first time I saw"
        assert result[0].messages[-1].role == Role.ASSISTANT


def test_infer_online_handles_invalid_content():
    """Test that the engine properly handles invalid content responses."""
    with aioresponses() as m:
        m.post(
            _TARGET_SERVER,
            status=200,
            body=json.dumps({"error": {"message": "Invalid JSON content"}}),
            content_type="application/json",
        )
        m.post(
            _TARGET_SERVER,
            status=200,
            body=json.dumps({"error": {"message": "Invalid JSON content"}}),
            content_type="application/json",
        )
        m.post(
            _TARGET_SERVER,
            status=200,
            body=json.dumps({"error": {"message": "Invalid JSON content"}}),
            content_type="application/json",
        )
        m.post(
            _TARGET_SERVER,
            status=200,
            body=json.dumps({"error": {"message": "Invalid JSON content"}}),
            content_type="application/json",
        )

        engine = RemoteInferenceEngine(
            model_params=_get_default_model_params(),
            remote_params=RemoteParams(
                api_url=_TARGET_SERVER,
                max_retries=2,
                retry_backoff_base=0.1,  # Small value for testing
                retry_backoff_max=0.3,
            ),
        )
        conversation = Conversation(
            messages=[
                Message(
                    role=Role.USER,
                    content="Hello world!",
                ),
            ],
        )

        inference_config = _get_default_inference_config()
        inference_config.remote_params = engine._remote_params

        async def mock_sleep(delay):
            pass

        with patch("asyncio.sleep", side_effect=mock_sleep):
            with pytest.raises(
                RuntimeError, match="Failed to process successful response"
            ):
                engine.infer(
                    [conversation],
                    inference_config,
                )


def test_infer_online_exponential_backoff():
    """Test that the engine implements exponential backoff correctly."""
    sleep_calls = []

    async def mock_sleep(delay):
        sleep_calls.append(delay)

    def callback(url, **kwargs):
        # Fail until the last attempt
        if len(sleep_calls) < 3:
            return CallbackResult(
                status=500,  # Use 500 instead of 401 since 401 is non-retriable
                body=json.dumps({"error": {"message": "Server Error"}}),
                content_type="application/json",
            )
        return CallbackResult(
            status=200,
            body=json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "Success after retries",
                            }
                        }
                    ]
                }
            ),
            content_type="application/json",
        )

    with aioresponses() as m:
        m.post(_TARGET_SERVER, callback=callback, repeat=True)

        with patch("asyncio.sleep", side_effect=mock_sleep):
            engine = RemoteInferenceEngine(
                model_params=_get_default_model_params(),
                remote_params=RemoteParams(
                    api_url=_TARGET_SERVER,
                    max_retries=2,
                    retry_backoff_base=0.2,  # Small values for testing
                    retry_backoff_max=1.0,
                ),
            )
            conversation = Conversation(
                messages=[Message(role=Role.USER, content="Hello")],
            )

            remote_params = RemoteParams(
                api_url=_TARGET_SERVER,
                max_retries=2,
                retry_backoff_base=0.2,
                retry_backoff_max=1.0,
            )
            inference_config = _get_default_inference_config()
            inference_config.remote_params = remote_params

            result = engine.infer([conversation], inference_config)

            # Verify the result
            assert len(result) == 1
            assert result[0].messages[-1].content == "Success after retries"

            # Verify sleep calls
            backoff_sleeps = [s for s in sleep_calls if s > 0]
            assert backoff_sleeps[0] == pytest.approx(0.2)  # First retry: base delay
            assert backoff_sleeps[1] == pytest.approx(
                0.4
            )  # Second retry: base delay * 2


def test_non_retriable_errors(mock_asyncio_sleep):
    """Test that certain HTTP status codes are not retried."""
    non_retriable_codes = [400, 401, 403, 404, 422]
    error_messages = {
        400: "Bad request error",
        401: "Unauthorized error",
        403: "Forbidden error",
        404: "Not found error",
        422: "Validation error",
    }

    for status_code in non_retriable_codes:
        with aioresponses() as m:
            m.post(
                _TARGET_SERVER,
                status=status_code,
                payload={"error": {"message": error_messages[status_code]}},
            )

            engine = RemoteInferenceEngine(
                model_params=_get_default_model_params(),
                remote_params=RemoteParams(
                    api_url=_TARGET_SERVER,
                    max_retries=3,
                ),
            )
            conversation = create_test_text_only_conversation()

            with pytest.raises(RuntimeError) as exc_info:
                engine.infer([conversation])

            assert f"Non-retriable error: {error_messages[status_code]}" in str(
                exc_info.value
            )
            # Verify no retries were attempted
            assert mock_asyncio_sleep.call_count == 0
            mock_asyncio_sleep.reset_mock()


def test_response_processing_error(mock_asyncio_sleep):
    """Test handling of errors during response processing."""
    with aioresponses() as m:
        m.post(
            _TARGET_SERVER,
            status=200,
            payload={"choices": [{"invalid": "response"}]},  # Missing required fields
        )
        m.post(
            _TARGET_SERVER,
            status=200,
            payload={"choices": [{"invalid": "response"}]},  # Missing required fields
        )
        m.post(
            _TARGET_SERVER,
            status=200,
            payload={"choices": [{"invalid": "response"}]},  # Missing required fields
        )
        m.post(
            _TARGET_SERVER,
            status=200,
            payload={"choices": [{"invalid": "response"}]},  # Missing required fields
        )

        engine = RemoteInferenceEngine(
            model_params=_get_default_model_params(),
            remote_params=RemoteParams(
                api_url=_TARGET_SERVER,
                max_retries=2,
            ),
        )
        conversation = create_test_text_only_conversation()

        with pytest.raises(RuntimeError) as exc_info:
            engine.infer([conversation])

        assert "Failed to process successful response" in str(exc_info.value)
        # Verify retries were attempted
        assert mock_asyncio_sleep.call_count == 4


def test_malformed_json_response(mock_asyncio_sleep):
    """Test handling of malformed JSON responses."""
    with aioresponses() as m:
        m.post(
            _TARGET_SERVER,
            status=200,
            body="Invalid JSON {",
            content_type="application/json",
        )
        m.post(
            _TARGET_SERVER,
            status=200,
            body="Invalid JSON {",
            content_type="application/json",
        )
        m.post(
            _TARGET_SERVER,
            status=200,
            body="Invalid JSON {",
            content_type="application/json",
        )

        engine = RemoteInferenceEngine(
            model_params=_get_default_model_params(),
            remote_params=RemoteParams(
                api_url=_TARGET_SERVER,
                max_retries=2,
            ),
        )
        conversation = create_test_text_only_conversation()

        with pytest.raises(RuntimeError) as exc_info:
            engine.infer([conversation])

        assert "Failed to parse response" in str(exc_info.value)
        assert "Content type: application/json" in str(exc_info.value)
        # Verify retries were attempted
        assert mock_asyncio_sleep.call_count == 4


def test_unexpected_error_handling(mock_asyncio_sleep):
    """Test handling of unexpected errors during API calls."""

    def raise_unexpected(*args, **kwargs):
        raise ValueError("Unexpected internal error")

    with aioresponses() as m:
        m.post(_TARGET_SERVER, callback=raise_unexpected)

        engine = RemoteInferenceEngine(
            model_params=_get_default_model_params(),
            remote_params=RemoteParams(
                api_url=_TARGET_SERVER,
                max_retries=2,
            ),
        )
        conversation = create_test_text_only_conversation()

        with pytest.raises(RuntimeError) as exc_info:
            engine.infer([conversation])

        assert (
            "Failed to query API after 3 attempts due to unexpected error: Unexpected "
            "internal error" in str(exc_info.value)
        )
        # Verify retries were attempted
        assert mock_asyncio_sleep.call_count == 4


def test_list_response_error_handling():
    """Test handling of list-type error responses."""
    with aioresponses() as m:
        m.post(
            _TARGET_SERVER,
            status=500,
            payload=[{"error": {"message": "Internal server error"}}],
        )

        engine = RemoteInferenceEngine(
            model_params=_get_default_model_params(),
            remote_params=RemoteParams(
                api_url=_TARGET_SERVER,
                max_retries=0,
            ),
        )
        conversation = create_test_text_only_conversation()

        with pytest.raises(RuntimeError) as exc_info:
            engine.infer([conversation])

        assert "Internal server error" in str(exc_info.value)


def test_retry_with_different_errors():
    """Test retry behavior with different types of errors on each attempt."""
    attempt = 0

    def get_response(*args, **kwargs):
        nonlocal attempt
        attempt += 1

        if attempt == 1:
            raise aiohttp.ClientError("Network error")
        elif attempt == 2:
            return CallbackResult(status=200, body="Invalid JSON {")
        elif attempt == 3:
            return CallbackResult(
                status=500, payload={"error": {"message": "Server error"}}
            )
        else:
            return CallbackResult(
                status=200,
                payload={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "Success after retries",
                            }
                        }
                    ]
                },
            )

    with aioresponses() as m:
        m.post(_TARGET_SERVER, callback=get_response)
        m.post(_TARGET_SERVER, callback=get_response)
        m.post(_TARGET_SERVER, callback=get_response)
        m.post(_TARGET_SERVER, callback=get_response)

        engine = RemoteInferenceEngine(
            model_params=_get_default_model_params(),
            remote_params=RemoteParams(
                api_url=_TARGET_SERVER,
                max_retries=3,
            ),
        )
        conversation = create_test_text_only_conversation()

        result = engine.infer([conversation])

        assert len(result) == 1
        assert result[0].messages[-1].content == "Success after retries"
        assert attempt == 4  # Verify all attempts were made


def test_adaptive_concurrency_initialization_enabled():
    """Test that adaptive concurrency controller is initialized when enabled."""
    remote_params = RemoteParams(
        api_url=_TARGET_SERVER,
        use_adaptive_concurrency=True,
        num_workers=20,
        politeness_policy=30.0,
    )

    engine = RemoteInferenceEngine(
        model_params=_get_default_model_params(),
        remote_params=remote_params,
    )

    # Should have adaptive concurrency controller
    assert hasattr(engine, "_adaptive_concurrency_controller")
    assert engine._adaptive_concurrency_controller is not None

    # Verify controller parameters are calculated correctly
    controller = engine._adaptive_concurrency_controller
    expected_min_concurrency = 1
    expected_max_concurrency = 20
    expected_initial_concurrency = 10
    expected_step = 2
    expected_min_update_time = 29.0

    assert controller._config.min_concurrency == expected_min_concurrency
    assert controller._config.max_concurrency == expected_max_concurrency
    assert controller._current_concurrency == expected_initial_concurrency
    assert controller._config.concurrency_step == expected_step
    assert controller._config.min_update_time == expected_min_update_time


def test_adaptive_concurrency_initialization_disabled():
    """Test that adaptive concurrency controller is not initialized when disabled."""
    remote_params = RemoteParams(
        api_url=_TARGET_SERVER,
        use_adaptive_concurrency=False,
        num_workers=20,
    )

    engine = RemoteInferenceEngine(
        model_params=_get_default_model_params(),
        remote_params=remote_params,
    )

    # Should not have adaptive concurrency controller
    assert not hasattr(engine, "_adaptive_concurrency_controller")


def test_adaptive_concurrency_parameter_edge_cases():
    """Test adaptive concurrency parameter calculations with edge cases."""
    # Test with very small num_workers
    remote_params = RemoteParams(
        api_url=_TARGET_SERVER,
        use_adaptive_concurrency=True,
        num_workers=2,
        politeness_policy=1.0,
    )

    engine = RemoteInferenceEngine(
        model_params=_get_default_model_params(),
        remote_params=remote_params,
    )

    controller = engine._adaptive_concurrency_controller
    assert controller._config.min_concurrency == 1
    assert controller._current_concurrency == 1
    assert controller._config.max_concurrency == 2
    assert controller._config.concurrency_step == 1
    assert controller._config.min_update_time == 1


def test_adaptive_concurrency_used_in_query_api():
    """Test that adaptive concurrency controller is used instead of semaphore."""
    with aioresponses() as m:
        m.post(
            _TARGET_SERVER,
            status=200,
            payload=dict(
                choices=[
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Test response",
                        }
                    }
                ]
            ),
        )

        remote_params = RemoteParams(
            api_url=_TARGET_SERVER,
            use_adaptive_concurrency=True,
            num_workers=10,
        )

        engine = RemoteInferenceEngine(
            model_params=_get_default_model_params(),
            remote_params=remote_params,
        )

        conversation = create_test_text_only_conversation()

        # Mock the adaptive concurrency controller
        mock_controller = AsyncContextManagerMock()
        mock_controller.record_success = AsyncMock()

        with patch.object(engine, "_adaptive_concurrency_controller", mock_controller):
            # This should use the adaptive concurrency controller instead of semaphore
            result = engine._infer_online([conversation])

            # Verify the conversation was processed
            assert len(result) == 1
            assert len(result[0].messages) == 5  # Original 4 + 1 response
            mock_controller.record_success.assert_called_once()
            mock_controller.record_error.assert_not_called()


def test_semaphore_used_when_adaptive_concurrency_disabled():
    """Test that semaphore is used when adaptive concurrency is disabled."""
    with aioresponses() as m:
        m.post(
            _TARGET_SERVER,
            status=200,
            payload=dict(
                choices=[
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Test response",
                        }
                    }
                ]
            ),
        )

        remote_params = RemoteParams(
            api_url=_TARGET_SERVER,
            use_adaptive_concurrency=False,
            num_workers=10,
        )

        engine = RemoteInferenceEngine(
            model_params=_get_default_model_params(),
            remote_params=remote_params,
        )

        conversation = create_test_text_only_conversation()

        # Should not have adaptive concurrency controller
        assert not hasattr(engine, "_adaptive_concurrency_controller")

        # This should use the regular semaphore
        result = engine._infer_online([conversation])

        # Verify the conversation was processed
        assert len(result) == 1
        assert len(result[0].messages) == 5  # Original 4 + 1 response


def test_adaptive_concurrency_success_recorded_on_successful_response():
    """Test that success is recorded when request succeeds with adaptive concurrency."""
    with aioresponses() as m:
        m.post(
            _TARGET_SERVER,
            status=200,
            payload=dict(
                choices=[
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Test response",
                        }
                    }
                ]
            ),
        )

        remote_params = RemoteParams(
            api_url=_TARGET_SERVER,
            use_adaptive_concurrency=True,
            num_workers=10,
        )

        engine = RemoteInferenceEngine(
            model_params=_get_default_model_params(),
            remote_params=remote_params,
        )

        conversation = create_test_text_only_conversation()

        with patch.object(engine, "_try_record_success") as mock_record_success:
            engine._infer_online([conversation])
            mock_record_success.assert_called_once()


def test_adaptive_concurrency_error_recorded_on_http_error():
    """Test that error is recorded when a request fails with HTTP error."""
    with aioresponses() as m:
        m.post(
            _TARGET_SERVER,
            status=500,
            payload={"error": {"message": "Internal server error"}},
        )

        remote_params = RemoteParams(
            api_url=_TARGET_SERVER,
            use_adaptive_concurrency=True,
            num_workers=10,
            max_retries=0,  # Don't retry so we fail quickly
        )

        engine = RemoteInferenceEngine(
            model_params=_get_default_model_params(),
            remote_params=remote_params,
        )

        conversation = create_test_text_only_conversation()

        with patch.object(engine, "_try_record_error") as mock_record_error:
            with pytest.raises(RuntimeError):
                engine._infer_online([conversation])
            # Should record error for each retry attempt
            assert mock_record_error.call_count >= 1


def test_adaptive_concurrency_error_recorded_on_json_parse_error():
    """Test that error is recorded when response JSON parsing fails."""
    with aioresponses() as m:
        m.post(
            _TARGET_SERVER,
            status=200,
            body="Invalid JSON response",
            content_type="application/json",
        )

        remote_params = RemoteParams(
            api_url=_TARGET_SERVER,
            use_adaptive_concurrency=True,
            num_workers=10,
            max_retries=0,  # Don't retry so we fail quickly
        )

        engine = RemoteInferenceEngine(
            model_params=_get_default_model_params(),
            remote_params=remote_params,
        )

        conversation = create_test_text_only_conversation()

        with patch.object(engine, "_try_record_error") as mock_record_error:
            with pytest.raises(RuntimeError):
                engine._infer_online([conversation])
            mock_record_error.assert_called()


@pytest.mark.asyncio
async def test_adaptive_concurrency_full_adjustment_cycle():
    """Test the complete adaptive concurrency cycle: warmup, backoff, and recovery."""
    attempt_count = 0
    time_progression = 0.0
    asserts_passed = True
    async_lock = asyncio.Lock()

    async def get_response(*args, **kwargs):
        async with async_lock:
            nonlocal attempt_count
            nonlocal time_progression
            nonlocal mock_time
            time_progression += 1.0
            mock_time.return_value = time_progression
            attempt_count += 1
            nonlocal controller
            nonlocal initial_concurrency
            nonlocal min_concurrency
            nonlocal max_concurrency
            nonlocal warmup_concurrency
            nonlocal backoff_concurrency
            nonlocal asserts_passed

            window_size = controller._config.min_window_size  # 10
            backoff_step = window_size * 2 + 1  # 21
            recovery_step = backoff_step + window_size * 2  # 41

            current_concurrency = controller._current_concurrency
            step_size = controller._config.concurrency_step

            # Steps 1-10: Concurrency should be 10
            # Step 11: Concurrency should be 12, window size 1
            # Step 12-20: Concurrency should be 12, window size 10
            # Step 21: Backoff to 10, window size 1
            # Step 22-30: Concurrency should be 10, window size 10
            # Step 31: Concurrency should stay 10, first good window
            # Step 32-40: Concurrency should stay 10, window size 10
            # Step 41: 2nd good window, exit backoff, warmup to 12, window size 1
            # Step 42-50: Concurrency should be 12, window size 10, finish before warmup

            try:
                # Phase 1: First window_size requests succeed with no warmup
                if attempt_count <= window_size:
                    assert current_concurrency == initial_concurrency
                    return CallbackResult(
                        status=200,
                        payload={
                            "choices": [
                                {
                                    "message": {
                                        "role": "assistant",
                                        "content": f"Success response {attempt_count}",
                                    }
                                }
                            ]
                        },
                    )
                elif attempt_count <= window_size * 2:
                    # Warmup happens once per window_size
                    assert current_concurrency == initial_concurrency + step_size
                    warmup_concurrency = current_concurrency
                    if attempt_count == window_size * 2:
                        warmup_concurrency = current_concurrency
                        backoff_concurrency = max(
                            math.floor(current_concurrency * 0.8), min_concurrency
                        )
                        return CallbackResult(
                            status=500,
                            payload={"error": {"message": "Server error"}},
                        )

                    return CallbackResult(
                        status=200,
                        payload={
                            "choices": [
                                {
                                    "message": {
                                        "role": "assistant",
                                        "content": f"Success response {attempt_count}",
                                    }
                                }
                            ]
                        },
                    )
                elif attempt_count < recovery_step:
                    # Concurrency should not change as we're waiting for 2 good windows
                    assert current_concurrency == backoff_concurrency
                    return CallbackResult(
                        status=200,
                        payload={
                            "choices": [
                                {
                                    "message": {
                                        "role": "assistant",
                                        "content": f"Good window {attempt_count}",
                                    }
                                }
                            ]
                        },
                    )
                elif attempt_count >= recovery_step:
                    # Two good windows have passed, so we should warmup again
                    assert current_concurrency == backoff_concurrency + step_size
                    return CallbackResult(
                        status=200,
                        payload={
                            "choices": [
                                {
                                    "message": {
                                        "role": "assistant",
                                        "content": f"Recovery response {attempt_count}",
                                    }
                                }
                            ]
                        },
                    )

            except AssertionError as e:
                print(f"AssertionError: {e}")
                asserts_passed = False
                raise

    with aioresponses() as m:
        m.post(_TARGET_SERVER, callback=get_response, repeat=True)

        remote_params = RemoteParams(
            api_url=_TARGET_SERVER,
            use_adaptive_concurrency=True,
            num_workers=20,
            max_retries=1,  # 1 retry to avoid complexity
        )

        engine = RemoteInferenceEngine(
            model_params=_get_default_model_params(),
            remote_params=remote_params,
        )

        controller = engine._adaptive_concurrency_controller
        assert controller._current_concurrency == 10
        min_concurrency = controller._config.min_concurrency
        max_concurrency = controller._config.max_concurrency
        initial_concurrency = controller._current_concurrency
        warmup_concurrency = -1
        backoff_concurrency = -1

        conversations = [
            Conversation(
                messages=[Message(content=f"Test message {i}", role=Role.USER)],
                conversation_id=f"conv-{i}",
            )
            for i in range(50)
        ]

        with patch(
            "oumi.inference.adaptive_concurrency_controller.time.time",
        ) as mock_time:
            result = engine.infer(conversations)
            assert len(result) == 50

        assert asserts_passed
